# 第 10 章 Retry 策略、错误处理与节点 Cache

在 LLM 应用中，网络调用终究会失败——API 限流、瞬时超时、模型服务抖动，每一次 LLM 调用都可能因为瞬态故障而中断。LangGraph 通过 `RetryPolicy` 提供了内建的重试机制，通过 `CachePolicy` 提供了节点级结果缓存，并通过一套精心设计的错误分类体系将 "可恢复" 与 "不可恢复" 的异常区分开来。

本章将深入 retry 和 cache 的源码实现，分析 `default_retry_on` 的黑名单策略、指数退避与 jitter 的计算逻辑、`CachePolicy` 的 key 生成机制，以及 `GraphInterrupt`、`GraphRecursionError` 等错误类型在引擎中的特殊地位。

## RetryPolicy：六个参数的精密控制

`RetryPolicy` 是一个 `NamedTuple`，定义了重试行为的所有维度：

```python
# langgraph/types.py
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5
    """第一次重试前的等待时间（秒）"""
    backoff_factor: float = 2.0
    """每次重试后间隔的乘数"""
    max_interval: float = 128.0
    """重试间隔的上限（秒）"""
    max_attempts: int = 3
    """最大尝试次数（包含首次执行）"""
    jitter: bool = True
    """是否在间隔中加入随机抖动"""
    retry_on: (
        type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool]
    ) = default_retry_on
    """哪些异常触发重试——可以是异常类、异常类列表，或判断函数"""
```

默认值构成了一个合理的配置：首次重试等 0.5 秒，之后每次翻倍（0.5 -> 1.0 -> 2.0 -> 4.0 ...），最大不超过 128 秒，总共尝试 3 次（1 次执行 + 2 次重试），并加入随机 jitter 防止多节点同时重试引发的 "惊群效应"。

### retry_on 的三种形态

`retry_on` 参数设计了极高的灵活性。`_should_retry_on()` 函数负责统一三种形态的判断逻辑：

```python
# langgraph/pregel/_retry.py
def _should_retry_on(retry_policy: RetryPolicy, exc: Exception) -> bool:
    if isinstance(retry_policy.retry_on, Sequence):
        return isinstance(exc, tuple(retry_policy.retry_on))
    elif isinstance(retry_policy.retry_on, type) and issubclass(
        retry_policy.retry_on, Exception
    ):
        return isinstance(exc, retry_policy.retry_on)
    elif callable(retry_policy.retry_on):
        return retry_policy.retry_on(exc)
    else:
        raise TypeError(
            "retry_on must be an Exception class, a list or tuple of "
            "Exception classes, or a callable"
        )
```

三种使用方式：

```python
# 方式一：异常类列表
RetryPolicy(retry_on=[httpx.TimeoutException, ConnectionError])

# 方式二：单个异常类
RetryPolicy(retry_on=httpx.HTTPStatusError)

# 方式三：自定义判断函数
RetryPolicy(retry_on=lambda exc: "rate limit" in str(exc).lower())
```

## default_retry_on：黑名单策略

默认的 `retry_on` 函数采用了一种 "默认重试，显式排除" 的黑名单策略：

```python
# langgraph/_internal/_retry.py
def default_retry_on(exc: Exception) -> bool:
    import httpx
    import requests

    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.HTTPError):
        return 500 <= exc.response.status_code < 600 if exc.response else True
    if isinstance(
        exc,
        (
            ValueError,
            TypeError,
            ArithmeticError,
            ImportError,
            LookupError,
            NameError,
            SyntaxError,
            RuntimeError,
            ReferenceError,
            StopIteration,
            StopAsyncIteration,
            OSError,
        ),
    ):
        return False
    return True
```

这个函数的设计哲学值得细读：

**优先处理的网络异常：**
- `ConnectionError`：无条件重试。网络连接失败几乎总是瞬态的
- `httpx.HTTPStatusError`：只有 5xx 才重试。4xx 是客户端错误（参数错、认证失败等），重试无意义
- `requests.HTTPError`：同上逻辑，但多了一个 `exc.response is None` 的保护——没有 response 对象时也重试

**显式排除的编程错误：**

| 异常类型 | 排除原因 |
|---|---|
| `ValueError` / `TypeError` | 参数错误，重试不会改变结果 |
| `ArithmeticError` | 数学错误（除零等），确定性失败 |
| `ImportError` | 缺少依赖包，不会自愈 |
| `LookupError` | KeyError / IndexError，数据结构问题 |
| `NameError` / `SyntaxError` | 代码错误，不可恢复 |
| `RuntimeError` | 通常是逻辑错误 |
| `ReferenceError` | 弱引用失效 |
| `StopIteration` / `StopAsyncIteration` | 迭代器耗尽 |
| `OSError` | 文件系统错误，通常不可恢复 |

**最后的兜底：`return True`**——对于所有未被上述规则匹配的异常（比如第三方库的自定义异常），默认选择重试。这是一个保守但务实的选择：未知异常有可能是瞬态的，宁可多试一次也不要轻易放弃。

注意 `httpx` 和 `requests` 是在函数体内 import 的——这是一个延迟导入模式，避免在不使用这些库的环境中引发 `ImportError`。

## run_with_retry：重试循环

同步版本的重试逻辑在 `run_with_retry()` 中实现。函数签名如下：

```python
# langgraph/pregel/_retry.py
def run_with_retry(
    task: PregelExecutableTask,
    retry_policy: Sequence[RetryPolicy] | None,
    configurable: dict[str, Any] | None = None,
) -> None:
```

函数内部是一个 `while True` 循环，每次迭代先清除 writes、执行 task，然后根据异常类型决定是冒泡、放弃还是重试。以下是关键的设计细节。

### 重试流程的六个关键细节

**1. 多策略匹配**

`retry_policy` 是一个 **序列**，而非单个策略。框架按顺序遍历所有 policy，找到第一个匹配当前异常的策略就使用它：

```python
matching_policy = None
for policy in retry_policy:
    if _should_retry_on(policy, exc):
        matching_policy = policy
        break
```

这允许为不同类型的异常配置不同的重试参数。例如，对限流错误使用更长的退避时间，对超时错误快速重试：

```python
builder.add_node(
    "api_call",
    call_api,
    retry=[
        # API 限流：长间隔、多次重试
        RetryPolicy(
            retry_on=[RateLimitError],
            initial_interval=5.0,
            max_attempts=10,
            backoff_factor=3.0,
        ),
        # 网络错误：短间隔、少量重试
        RetryPolicy(
            retry_on=[ConnectionError, TimeoutError],
            initial_interval=0.5,
            max_attempts=3,
        ),
    ],
)
```

**2. writes 清除**

每次重试前，`task.writes.clear()` 清除上一次尝试产生的所有 channel write。这确保了重试的 "原子性" 语义——一个 node 的 side effect 要么全部生效（成功），要么全部丢弃（失败重试）。如果不清除，失败的节点可能已经写入了部分数据到 channel 中，重试时这些部分数据会与新的输出混合，导致状态不一致。

**3. 指数退避计算**

```python
interval = matching_policy.initial_interval
interval = min(
    matching_policy.max_interval,
    interval * (matching_policy.backoff_factor ** (attempts - 1)),
)
```

以默认参数为例：

| 重试次数 | 基础间隔 | 加 jitter 后约 |
|----------|----------|----------------|
| 第 1 次 | 0.5 秒 | 0.5 ~ 1.5 秒 |
| 第 2 次 | 1.0 秒 | 1.0 ~ 2.0 秒 |
| 第 3 次 | 2.0 秒 | 2.0 ~ 3.0 秒 |
| 第 4 次 | 4.0 秒 | 4.0 ~ 5.0 秒 |
| ... | ... | ... |
| 第 9 次 | 128.0 秒 | 128.0 ~ 129.0 秒（达到上限） |

**4. jitter 实现**

```python
sleep_time = interval + random.uniform(0, 1) if matching_policy.jitter else interval
```

jitter 的实现非常简洁——在计算出的 interval 上加一个 `[0, 1)` 的随机数。这是一种 "additive jitter" 策略，足以有效打散多个并发 task 的重试时刻，避免惊群效应。

**5. 不可重试的异常类型**

两种异常被显式排除在重试之外：

- `GraphBubbleUp`（包括 `GraphInterrupt`）：中断不是错误，而是 human-in-the-loop 的控制流信号
- `ParentCommand`：子图向父图发送的 Command，需要冒泡而非重试

**6. CONFIG_KEY_RESUMING 信号**

重试时设置 `CONFIG_KEY_RESUMING: True`，通知子图这是一次恢复执行而非首次执行。这在子图有 checkpoint 的场景下非常重要——子图可以从上次中断处继续而非从头开始。

## 异步重试的额外能力

`arun_with_retry()` 在结构上与同步版本几乎相同，但多了三点差异：

1. **stream 参数**：当 `stream=True` 时使用 `task.proc.astream()` 而非 `ainvoke()`，确保流式回调正常触发
2. **cache 检查**：`match_cached_writes` 参数允许在执行前检查缓存——如果当前 task 已有缓存结果，直接 return 跳过执行
3. **sleep 方式**：使用 `await asyncio.sleep()` 替代 `time.sleep()`，避免阻塞事件循环

## 异常注释（Python 3.11+）

在 Python 3.11 及以上版本中，重试失败后抛出的异常会附带一个注释，包含 task 的名称和 ID：

```python
SUPPORTS_EXC_NOTES = sys.version_info >= (3, 11)

# 在重试逻辑中：
if SUPPORTS_EXC_NOTES:
    exc.add_note(f"During task with name '{task.name}' and id '{task.id}'")
```

这利用了 Python 3.11 引入的 PEP 678 `BaseException.add_note()` 方法，让 traceback 信息更加丰富，方便在多节点图中定位问题出在哪个 node。

## CachePolicy：节点结果缓存

`CachePolicy` 用于缓存节点的执行结果，当输入不变时直接返回缓存值而非重新执行：

```python
# langgraph/types.py
@dataclass(**_DC_KWARGS)
class CachePolicy(Generic[KeyFuncT]):
    key_func: KeyFuncT = default_cache_key
    """从节点输入生成缓存 key 的函数，默认使用 pickle 序列化"""

    ttl: int | None = None
    """缓存条目的过期时间（秒），None 表示永不过期"""
```

### default_cache_key：基于 pickle 的 key 生成

缓存 key 的生成分为两步——先 "冻结" 输入使其可 hash，再用 pickle 序列化为字节串：

```python
# langgraph/_internal/_cache.py
def _freeze(obj: Any, depth: int = 10) -> Hashable:
    if isinstance(obj, Hashable) or depth <= 0:
        return obj
    elif isinstance(obj, Mapping):
        # 排序 key，确保 {"a":1,"b":2} == {"b":2,"a":1}
        return tuple(sorted((k, _freeze(v, depth - 1)) for k, v in obj.items()))
    elif isinstance(obj, Sequence):
        return tuple(_freeze(x, depth - 1) for x in obj)
    elif hasattr(obj, "tobytes"):
        # numpy/pandas 等对象
        return (
            type(obj).__name__,
            obj.tobytes(),
            obj.shape if hasattr(obj, "shape") else None,
        )
    return obj

def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    import pickle
    return pickle.dumps(
        (_freeze(args), _freeze(kwargs)),
        protocol=5,
        fix_imports=False,
    )
```

`_freeze` 的设计细节：

1. **Mapping 排序**：`tuple(sorted(...))` 确保字典的 key 顺序不影响缓存 key，`{"a":1,"b":2}` 和 `{"b":2,"a":1}` 产生相同的冻结结果
2. **Sequence 转换**：列表转为 tuple，使其可 hash
3. **科学计算支持**：通过 `tobytes()` 方法处理 numpy 数组等对象，同时保留 shape 信息
4. **深度限制**：递归深度限制为 10，防止深嵌套对象无限递归
5. **pickle protocol 5**：Python 3.8+ 引入，在速度和空间之间取得平衡。`fix_imports=False` 禁用 Python 2 兼容性，略微提升性能

### CacheKey 与 PregelExecutableTask

缓存 key 在 task 创建阶段就被计算好，存储在 `PregelExecutableTask.cache_key` 字段中：

```python
# langgraph/types.py
class CacheKey(NamedTuple):
    ns: tuple[str, ...]
    """缓存条目的 namespace"""
    key: str
    """由 key_func 生成的缓存 key"""
    ttl: int | None
    """过期时间（秒）"""

@dataclass(**_T_DC_KWARGS)
class PregelExecutableTask:
    name: str
    input: Any
    proc: Runnable
    writes: deque[tuple[str, Any]]
    config: RunnableConfig
    triggers: Sequence[str]
    retry_policy: Sequence[RetryPolicy]
    cache_key: CacheKey | None      # <-- 缓存 key
    id: str
    path: tuple[str | int | tuple, ...]
    writers: Sequence[Runnable] = ()
    subgraphs: Sequence[PregelProtocol] = ()
```

`CacheKey` 是一个 `NamedTuple`，三个字段组合唯一标识一个缓存条目。`ns` 用于区分不同（子）图的缓存空间，避免不同子图中同名节点的缓存冲突。

### 缓存与 stream 输出

在主循环中，缓存命中的 task 有特殊的输出处理：

```python
# langgraph/pregel/main.py (stream 方法 loop 内)
while loop.tick():
    for task in loop.match_cached_writes():
        loop.output_writes(task.id, task.writes, cached=True)
```

`cached=True` 标志影响了 stream 输出——缓存命中的 task 不会发送 `tasks` 模式的事件（因为它并没有真正 "执行"），但会正常发送 `updates` 事件，保持输出的一致性。

## ErrorCode：结构化的错误分类

LangGraph 定义了一组 `ErrorCode` 枚举，每种错误都有对应的在线文档：

```python
# langgraph/errors.py
class ErrorCode(Enum):
    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"

def create_error_message(*, message: str, error_code: ErrorCode) -> str:
    return (
        f"{message}\n"
        "For troubleshooting, visit: "
        "https://docs.langchain.com/oss/python/langgraph/"
        f"errors/{error_code.value}"
    )
```

`create_error_message` 在错误信息末尾附加了一个 URL，指向该错误码的在线排查指南。这是优秀的 DX（Developer Experience）设计——开发者在看到异常时可以直接点击链接获取帮助。

## 异常层次结构

LangGraph 的异常体系建立在一个核心概念上：**有些 "异常" 不是错误，而是控制流信号**。

```python
# langgraph/errors.py
class GraphBubbleUp(Exception):
    pass

class GraphInterrupt(GraphBubbleUp):
    """Raised when a subgraph is interrupted, suppressed by the root graph.
    Never raised directly, or surfaced to the user."""
    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)

class ParentCommand(GraphBubbleUp):
    args: tuple[Command]
    def __init__(self, command: Command) -> None:
        super().__init__(command)
```

继承关系：

```
Exception
  └── GraphBubbleUp          ← 控制流信号基类
  │     ├── GraphInterrupt   ← interrupt() 触发的中断
  │     │     └── NodeInterrupt (deprecated)
  │     └── ParentCommand    ← 子图向父图发命令
  ├── RecursionError
  │     └── GraphRecursionError  ← step 数量超限
  ├── InvalidUpdateError     ← 非法 channel 更新
  ├── EmptyInputError        ← 空输入
  └── TaskNotFound           ← 分布式模式找不到 task
```

### GraphBubbleUp：冒泡基类

`GraphBubbleUp` 是所有 "需要向上传播但不应触发重试" 的异常的基类。在 retry 逻辑中，它被优先捕获并直接 re-raise：

```python
# langgraph/pregel/_retry.py (run_with_retry 中)
except GraphBubbleUp:
    raise
```

这行代码位于 `except Exception` 之前，确保 `GraphInterrupt` 和 `ParentCommand` 永远不会进入重试判断。

### GraphInterrupt vs GraphRecursionError

这两个异常经常被初学者混淆，但它们的性质完全不同：

| 特性 | GraphInterrupt | GraphRecursionError |
|------|---------------|-------------------|
| 基类 | `GraphBubbleUp` | `RecursionError` |
| 性质 | 控制流信号，不是错误 | 真正的错误 |
| 是否可恢复 | 是（通过 `Command(resume=...)`） | 否（需提高 `recursion_limit`） |
| 是否触发重试 | 否 | 否（原因不同） |
| 典型场景 | human-in-the-loop 交互 | 图中存在无限循环 |
| 是否需要 checkpointer | 是 | 不一定 |

`GraphInterrupt` 通过 `interrupt()` 函数触发，携带一组 `Interrupt` 对象：

```python
# langgraph/types.py (interrupt 函数末尾)
raise GraphInterrupt(
    (
        Interrupt.from_ns(
            value=value,
            ns=conf[CONFIG_KEY_CHECKPOINT_NS],
        ),
    )
)
```

`GraphRecursionError` 则在主循环退出后由 `stream()` / `astream()` 方法检测 `loop.status == "out_of_steps"` 并抛出，错误消息中附带 `ErrorCode.GRAPH_RECURSION_LIMIT` 的文档链接。注意 LangGraph 的 "recursion limit" 与 Python 的函数调用栈深度无关——它限制的是 Pregel 模型中的 **step 数量**，默认值为 25。

### ParentCommand：跨图通信

`ParentCommand` 是子图向父图发送 `Command` 的机制。在 retry 逻辑中有三条路径：如果 Command 的目标 graph 与当前图匹配，就地处理并 break；如果目标是 `Command.PARENT`，通过 `_checkpoint_ns_for_parent_command` 重写 namespace 后继续冒泡；否则直接 raise。

`_checkpoint_ns_for_parent_command` 从 checkpoint namespace（格式 `parent_name:task_id|child_name:task_id`）中层层剥离当前 frame 和数字段，找到父图的 namespace。

## InvalidUpdateError 与其他异常

除异常层次结构外，LangGraph 还定义了几个独立的异常类：

- **`InvalidUpdateError`**：最常见的用户可见错误之一。多个并发 node 对同一个非聚合 channel 写入不同值（`INVALID_CONCURRENT_GRAPH_UPDATE`），或 node 返回了无法解析的值（`INVALID_GRAPH_NODE_RETURN_VALUE`），都会触发此异常
- **`EmptyInputError`**：图收到空输入时抛出
- **`TaskNotFound`**：分布式执行模式下，executor 无法在集群中找到指定 task 时抛出

## NodeInterrupt：已废弃的中断方式

`NodeInterrupt` 继承自 `GraphInterrupt`，是早期版本中用异常类实现中断的方式，现已被函数式的 `interrupt()` 取代（标记为 `LangGraphDeprecatedSinceV10`）。新版 `interrupt()` 支持 resume 值匹配、多次中断等更复杂的交互模式。

## RetryPolicy 在 Pregel 中的两层配置

重试策略可以在两个层级配置——图级别（`Pregel.retry_policy`）和节点级别（`PregelExecutableTask.retry_policy`）。在 `run_with_retry` 中，节点级策略优先：

```python
retry_policy = task.retry_policy or retry_policy
```

这使得对延迟敏感的节点可以配置快速重试，而调用外部 API 的节点可以使用更保守的退避时间。

## Retry 与 Cache 的协作

重试和缓存在 `arun_with_retry` 中有一个精妙的交互点。在异步版本中，执行前会先检查缓存：

```python
# langgraph/pregel/_retry.py (arun_with_retry)
if match_cached_writes is not None and task.cache_key is not None:
    for t in await match_cached_writes():
        if t is task:
            return
```

这意味着：如果一个 task 在上一次执行中已经成功并被缓存了，即使它被安排重新执行（例如由于其他 task 的失败导致整个 step 重跑），也会直接使用缓存结果。这避免了不必要的重复计算和 API 调用。

在 loop 层面，缓存命中的 write 通过 `cached=True` 标记：

```python
# langgraph/pregel/main.py (stream 方法 loop 内)
while loop.tick():
    for task in loop.match_cached_writes():
        loop.output_writes(task.id, task.writes, cached=True)
```

`output_writes` 方法对 `cached=True` 的 task 跳过 `tasks` 模式事件的发射：

```python
# langgraph/pregel/_loop.py (output_writes 末尾)
if not cached:
    self._emit(
        "tasks",
        map_debug_task_results,
        (task, writes),
        self.stream_keys,
    )
```

这确保了 `tasks` 和 `debug` stream mode 只报告真正执行过的 task，而 `updates` 和 `values` 则不受影响——缓存命中的结果同样需要反映在 state 更新中。

## 本章要点

1. **RetryPolicy** 通过六个参数提供了完整的重试控制：`initial_interval`、`backoff_factor`、`max_interval` 控制指数退避，`jitter` 防止惊群，`max_attempts` 设上限，`retry_on` 决定哪些异常触发重试

2. **default_retry_on** 采用黑名单策略——显式排除 `ValueError`、`TypeError` 等编程错误，对未知异常默认重试。HTTP 错误只重试 5xx，4xx 直接放弃

3. **多策略匹配**：`retry_policy` 是一个序列，按顺序遍历找到第一个匹配的策略，允许为不同异常类型配置不同的退避参数

4. **CachePolicy** 通过 `default_cache_key`（`_freeze` 递归冻结 + pickle 序列化）生成确定性的缓存 key，支持 TTL 过期。`_freeze` 对 dict 排序确保 key 顺序无关性

5. **异常层次**：`GraphBubbleUp` 家族（`GraphInterrupt`、`ParentCommand`）是控制流信号而非错误，永远不触发重试。`GraphRecursionError` 限制的是 Pregel step 数而非调用栈深度

6. **writes 清除**（`task.writes.clear()`）是重试原子性的保障——每次重试前丢弃上一次的部分写入，确保 state 一致性

7. **Retry 与 Cache 协作**：缓存命中的 task 跳过执行但保留 `updates` 输出；`tasks`/`debug` stream 事件只报告真正执行的 task，不报告缓存命中
