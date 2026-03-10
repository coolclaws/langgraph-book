# 第 17 章 Functional API：@entrypoint 与 @task

前面的章节中，我们深入分析了 `StateGraph` 这种声明式的图定义方式。LangGraph 还提供了一种函数式的替代方案 -- Functional API，通过 `@entrypoint` 和 `@task` 两个装饰器实现。本章将完整剖析 `langgraph/func/__init__.py` 的 575 行实现，理解 Functional API 如何在保持 Pregel 引擎能力的同时，提供更直观的编程模型。

## @task 装饰器

`@task` 是 Functional API 中定义可执行单元的方式。它将一个普通函数包装为一个可以被 LangGraph 引擎管理的 task，支持 retry、cache、以及 checkpoint 集成。

### _TaskFunction 包装类

```python
# libs/langgraph/langgraph/func/__init__.py
class _TaskFunction(Generic[P, T]):
    def __init__(
        self,
        func: Callable[P, Awaitable[T]] | Callable[P, T],
        *,
        retry_policy: Sequence[RetryPolicy],
        cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
        name: str | None = None,
    ) -> None:
        if name is not None:
            if hasattr(func, "__func__"):
                instance_method = functools.partial(func.__func__, func.__self__)
                instance_method.__name__ = name
                func = instance_method
            else:
                func.__name__ = name
        self.func = func
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        functools.update_wrapper(self, func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> SyncAsyncFuture[T]:
        return call(
            self.func,
            retry_policy=self.retry_policy,
            cache_policy=self.cache_policy,
            *args,
            **kwargs,
        )
```

核心设计点：

1. **`__call__` 返回 Future**：调用 `@task` 装饰的函数不会立即执行，而是返回一个 `SyncAsyncFuture[T]`。这使得并发执行成为可能。
2. **name 覆盖**：支持通过参数指定 task 名称，覆盖函数的原始名称。对实例方法做了特殊处理，通过 `functools.partial` 包装避免修改原始类方法。
3. **CachePolicy 与 RetryPolicy**：直接存储在 `_TaskFunction` 上，在调用时传递给引擎。

### task 装饰器的多重重载

```python
# libs/langgraph/langgraph/func/__init__.py
@overload
def task(
    __func_or_none__: None = None,
    *,
    name: str | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
) -> Callable[
    [Callable[P, Awaitable[T]] | Callable[P, T]],
    _TaskFunction[P, T],
]: ...

@overload
def task(__func_or_none__: Callable[P, Awaitable[T]]) -> _TaskFunction[P, T]: ...

@overload
def task(__func_or_none__: Callable[P, T]) -> _TaskFunction[P, T]: ...
```

三种 overload 支持三种使用方式：

```python
# 方式一：无参装饰器
@task
def add_one(a: int) -> int:
    return a + 1

# 方式二：带参装饰器
@task(retry_policy=RetryPolicy(max_attempts=3))
def risky_operation(data: str) -> str:
    return call_external_api(data)

# 方式三：异步函数
@task
async def async_add_one(a: int) -> int:
    return a + 1
```

实现的核心逻辑很简洁：

```python
# libs/langgraph/langgraph/func/__init__.py
def task(__func_or_none__=None, *, name=None, retry_policy=None, cache_policy=None):
    retry_policies: Sequence[RetryPolicy] = (
        ()
        if retry_policy is None
        else (retry_policy,)
        if isinstance(retry_policy, RetryPolicy)
        else retry_policy
    )

    def decorator(func):
        return _TaskFunction(
            func, retry_policy=retry_policies, cache_policy=cache_policy, name=name
        )

    if __func_or_none__ is not None:
        return decorator(__func_or_none__)

    return decorator
```

### call() 函数：task 的执行入口

`_TaskFunction.__call__` 调用的 `call()` 函数定义在 `_call.py` 中：

```python
# libs/langgraph/langgraph/pregel/_call.py
def call(
    func: Callable[P, Awaitable[T]] | Callable[P, T],
    *args: Any,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    **kwargs: Any,
) -> SyncAsyncFuture[T]:
    config = get_config()
    impl = config[CONF][CONFIG_KEY_CALL]
    fut = impl(
        func,
        (args, kwargs),
        retry_policy=retry_policy,
        cache_policy=cache_policy,
        callbacks=config["callbacks"],
    )
    return fut
```

`CONFIG_KEY_CALL` 是注入到 config 中的实际执行器。它负责：
1. 将函数和参数打包为一个 task
2. 通过引擎调度执行
3. 返回一个 Future，允许调用者决定何时获取结果

`SyncAsyncFuture` 同时支持同步和异步等待：

```python
# libs/langgraph/langgraph/pregel/_call.py
class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self) -> Generator[T, None, T]:
        yield cast(T, ...)
```

它继承了 `concurrent.futures.Future`（提供 `.result()` 方法），同时实现了 `__await__` 协议（支持 `await`）。

### task 的 Runnable 包装

在引擎内部，task 函数被包装为 `RunnableSeq`，最后会附加一个 `ChannelWrite` 来写入 RETURN channel：

```python
# libs/langgraph/langgraph/pregel/_call.py
def get_runnable_for_task(func: Callable[..., Any]) -> Runnable:
    # ...
    seq = RunnableSeq(
        run,
        ChannelWrite([ChannelWriteEntry(RETURN)]),
        name=name,
        trace_inputs=functools.partial(
            _explode_args_trace_inputs, inspect.signature(func)
        ),
    )
    return seq
```

RETURN channel 用于记录 task 的返回值，使得 checkpointer 可以在恢复时跳过已完成的 task。

## @entrypoint 装饰器

`@entrypoint` 定义工作流的入口点。它是一个类，而不是普通函数，这样设计是为了支持 `entrypoint.final` 属性。

### entrypoint 类结构

```python
# libs/langgraph/langgraph/func/__init__.py
class entrypoint(Generic[ContextT]):
    def __init__(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        cache: BaseCache | None = None,
        context_schema: type[ContextT] | None = None,
        cache_policy: CachePolicy | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.store = store
        self.cache = cache
        self.cache_policy = cache_policy
        self.retry_policy = retry_policy
        self.context_schema = context_schema
```

`entrypoint` 接受 checkpointer、store、cache 等参数 -- 这些与 `StateGraph.compile()` 的参数完全对应。

### entrypoint.final：分离返回值与持久化值

```python
# libs/langgraph/langgraph/func/__init__.py
@dataclass(**_DC_KWARGS)
class final(Generic[R, S]):
    value: R
    """Value to return. A value will always be returned even if it is None."""
    save: S
    """The value for the state for the next checkpoint."""
```

`entrypoint.final` 允许函数返回一个值给调用者，同时保存另一个值到 checkpoint。这在需要累积状态的场景中非常有用：

```python
@entrypoint(checkpointer=InMemorySaver())
def counter(increment: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    new_total = previous + increment
    # 返回 new_total 给调用者，保存 new_total 到 checkpoint
    return entrypoint.final(value=new_total, save=new_total)
```

### __call__：将函数转换为 Pregel 图

`entrypoint.__call__` 是最核心的方法，它将装饰的函数转换为一个完整的 Pregel 图：

```python
# libs/langgraph/langgraph/func/__init__.py
def __call__(self, func: Callable[..., Any]) -> Pregel:
    if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
        raise NotImplementedError(
            "Generators are not supported in the Functional API."
        )

    bound = get_runnable_for_entrypoint(func)
    stream_mode: StreamMode = "updates"

    sig = inspect.signature(func)
    first_parameter_name = next(iter(sig.parameters.keys()), None)
    if not first_parameter_name:
        raise ValueError("Entrypoint function must have at least one parameter")
    input_type = (
        sig.parameters[first_parameter_name].annotation
        if sig.parameters[first_parameter_name].annotation
        is not inspect.Signature.empty
        else Any
    )
```

然后定义两个辅助函数用于处理 `entrypoint.final`：

```python
    def _pluck_return_value(value: Any) -> Any:
        return value.value if isinstance(value, entrypoint.final) else value

    def _pluck_save_value(value: Any) -> Any:
        return value.save if isinstance(value, entrypoint.final) else value
```

最后构建 Pregel 图：

```python
    graph: Pregel = Pregel(
        nodes={
            func.__name__: PregelNode(
                bound=bound,
                triggers=[START],
                channels=START,
                writers=[
                    ChannelWrite(
                        [
                            ChannelWriteEntry(END, mapper=_pluck_return_value),
                            ChannelWriteEntry(PREVIOUS, mapper=_pluck_save_value),
                        ]
                    )
                ],
            )
        },
        channels={
            START: EphemeralValue(input_type),
            END: LastValue(output_type, END),
            PREVIOUS: LastValue(save_type, PREVIOUS),
        },
        input_channels=START,
        output_channels=END,
        stream_channels=END,
        stream_mode=stream_mode,
        stream_eager=True,
        checkpointer=self.checkpointer,
        store=self.store,
        cache=self.cache,
        cache_policy=self.cache_policy,
        retry_policy=self.retry_policy or (),
        context_schema=self.context_schema,
    )
    return graph
```

这段代码揭示了 Functional API 的本质：**它是 StateGraph 的语法糖**。一个 `@entrypoint` 装饰的函数被转换为一个只有一个节点的 Pregel 图，具有三个 channel：

| Channel | 类型 | 作用 |
|---------|------|------|
| `START` | `EphemeralValue` | 接收函数输入 |
| `END` | `LastValue` | 存储函数返回值 |
| `PREVIOUS` | `LastValue` | 存储 checkpoint 持久化值 |

节点通过 `ChannelWrite` 将函数返回值分别写入 `END` 和 `PREVIOUS` channel。如果返回的是 `entrypoint.final` 对象，则 `_pluck_return_value` 提取 `value` 写入 END，`_pluck_save_value` 提取 `save` 写入 PREVIOUS。

### get_runnable_for_entrypoint：函数到 Runnable 的转换

```python
# libs/langgraph/langgraph/pregel/_call.py
def get_runnable_for_entrypoint(func: Callable[..., Any]) -> Runnable:
    key = (func, False)
    if key in CACHE:
        return CACHE[key]
    else:
        if is_async_callable(func):
            run = RunnableCallable(
                None, func, name=func.__name__, trace=False, recurse=False
            )
        else:
            afunc = functools.update_wrapper(
                functools.partial(run_in_executor, None, func), func
            )
            run = RunnableCallable(
                func, afunc,
                name=func.__name__, trace=False, recurse=False,
            )
        if not _lookup_module_and_qualname(func):
            return run
        return CACHE.setdefault(key, run)
```

对于同步函数，LangGraph 会自动创建一个异步版本（通过 `run_in_executor`），使得 Pregel 引擎可以统一使用异步调度。`CACHE` 字典确保相同函数不会重复创建 Runnable。

## 与 StateGraph 的区别

### 声明式 vs 函数式

StateGraph 是声明式的：你先定义节点和边，然后编译。控制流由图的结构决定。

```python
# StateGraph 方式
builder = StateGraph(State)
builder.add_node("step1", fn1)
builder.add_node("step2", fn2)
builder.add_edge("step1", "step2")
graph = builder.compile()
```

Functional API 是函数式的：你写一个普通函数，控制流就是函数的控制流。

```python
# Functional API 方式
@entrypoint(checkpointer=InMemorySaver())
def workflow(input_data: str) -> str:
    result1 = step1_task(input_data).result()
    result2 = step2_task(result1).result()
    return result2
```

### 状态管理的差异

StateGraph 通过 channels 管理状态，每个节点读写 channel。状态在每一步都会被持久化。

Functional API 的状态就是函数的局部变量和 `previous` 参数。只有在函数返回时才会持久化（写入 PREVIOUS channel）。task 的结果通过 RETURN channel 被 checkpoint 记录，恢复时可以跳过已完成的 task。

### 路由逻辑的差异

StateGraph 使用 `add_conditional_edges` 和 `Command(goto=...)` 实现路由。Functional API 使用普通的 `if/else` 和函数调用。

## @task 的并发执行

Functional API 的一大优势是自然的并发支持。由于 `@task` 返回 Future，你可以轻松实现并行执行：

### 同步并发

```python
@task
def process_item(item: str) -> str:
    return heavy_computation(item)

@entrypoint()
def parallel_workflow(items: list[str]) -> list[str]:
    # 同时提交所有 task
    futures = [process_item(item) for item in items]
    # 收集结果
    results = [f.result() for f in futures]
    return results
```

### 异步并发（asyncio.gather 模式）

```python
@task
async def async_process(item: str) -> str:
    return await async_heavy_computation(item)

@entrypoint()
async def async_parallel(items: list[str]) -> list[str]:
    futures = [async_process(item) for item in items]
    return await asyncio.gather(*futures)
```

在异步模式下，`SyncAsyncFuture.__await__` 的实现使得 `asyncio.gather` 可以正确工作。引擎在底层调度这些 task 时，会利用 Pregel 的并行执行能力。

## Checkpoint 集成

`@entrypoint` 自动管理状态的持久化。当配置了 checkpointer 时：

1. **previous 参数**：函数可以声明一个 `previous` 参数来接收上次执行的保存值
2. **task 结果缓存**：已完成的 task 不会在恢复时重新执行
3. **interrupt 支持**：可以在 entrypoint 函数或 task 中使用 `interrupt()`

```python
@entrypoint(checkpointer=InMemorySaver())
def review_workflow(topic: str) -> dict:
    essay_future = compose_essay(topic)
    essay = essay_future.result()
    human_review = interrupt({
        "question": "Please provide a review",
        "essay": essay
    })
    return {
        "essay": essay,
        "review": human_review,
    }
```

恢复时，`compose_essay` 的结果从 checkpoint 中恢复（不会重新执行），而 `interrupt()` 返回用户提供的 resume 值。

## 可注入参数详解

entrypoint 函数除了必须的第一个输入参数外，还支持几种注入参数：

### previous 参数

```python
@entrypoint(checkpointer=InMemorySaver())
def my_workflow(
    input_data: str,
    previous: Optional[str] = None
) -> str:
    if previous:
        return f"续接: {previous} -> {input_data}"
    return f"首次: {input_data}"

config = {"configurable": {"thread_id": "thread-1"}}

my_workflow.invoke("hello", config)   # "首次: hello"
my_workflow.invoke("world", config)   # "续接: hello -> world"
```

`previous` 的值来自 `PREVIOUS` channel -- 即上次执行中写入 checkpoint 的值。这是 Functional API 实现跨调用状态持续的核心机制。

PREVIOUS channel 在 Pregel 图中的定义：

```python
# libs/langgraph/langgraph/_internal/_constants.py
PREVIOUS = sys.intern("__previous__")
```

```python
# entrypoint.__call__ 中
channels={
    # ...
    PREVIOUS: LastValue(save_type, PREVIOUS),
}
```

### config 参数

```python
@entrypoint()
def my_workflow(input_data: str, config: RunnableConfig) -> str:
    thread_id = config["configurable"]["thread_id"]
    return f"Running on thread: {thread_id}"
```

### runtime 参数

```python
@entrypoint(store=my_store)
def my_workflow(input_data: str, runtime: Runtime) -> str:
    # 访问 store
    items = runtime.store.search(("namespace",))
    # 使用 stream writer
    runtime.writer("intermediate result")
    return "done"
```

## entrypoint.final 的类型推导

entrypoint 的 `__call__` 方法中有一段精细的返回类型推导逻辑：

```python
# libs/langgraph/langgraph/func/__init__.py
output_type, save_type = Any, Any
if sig.return_annotation is not inspect.Signature.empty:
    if sig.return_annotation is entrypoint.final:
        # 未参数化的 entrypoint.final，两者都是 Any
        output_type = save_type = Any
    else:
        origin = get_origin(sig.return_annotation)
        if origin is entrypoint.final:
            type_annotations = get_args(sig.return_annotation)
            if len(type_annotations) != 2:
                raise TypeError(
                    "Please an annotation for both the return_ and "
                    "the save values."
                )
            output_type, save_type = get_args(sig.return_annotation)
        else:
            # 普通返回类型：output 和 save 相同
            output_type = save_type = sig.return_annotation
```

三种情况：
1. **`-> entrypoint.final`**（未参数化）：`output_type = save_type = Any`
2. **`-> entrypoint.final[int, str]`**（参数化）：`output_type = int`, `save_type = str`
3. **`-> int`**（普通类型）：`output_type = save_type = int`

## 不支持生成器函数

```python
if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
    raise NotImplementedError(
        "Generators are not supported in the Functional API."
    )
```

如果需要流式输出中间结果，应使用 `StreamWriter`（通过 `runtime.writer` 注入），而非 Python 的 yield 语法。

## Serde Allowlist：严格序列化白名单

在启用严格序列化模式（`STRICT_MSGPACK_ENABLED`）时，Functional API 会构建一个类型白名单：

```python
# libs/langgraph/langgraph/func/__init__.py
if _serde.STRICT_MSGPACK_ENABLED:
    serde_allowlist = _serde.build_serde_allowlist(
        schemas=[input_type, output_type, save_type]
        + ([self.context_schema] if self.context_schema is not None else []),
        channels=graph.channels,
    )
    graph._serde_allowlist = serde_allowlist
    graph.checkpointer = _serde.apply_checkpointer_allowlist(
        graph.checkpointer, serde_allowlist
    )
```

白名单基于函数签名中的类型注解（input_type、output_type、save_type）和 context_schema 生成。只有白名单中的类型可以被序列化/反序列化，防止意外的对象类型被 checkpoint 系统处理。

## stream_eager=True 的含义

在 Pregel 构建中，`stream_eager=True` 被设置：

```python
graph = Pregel(
    # ...
    stream_eager=True,
    # ...
)
```

这意味着 Functional API 的图在执行时会急切地发射 stream 事件，而不是等到整个 superstep 完成。对于长时间运行的 entrypoint（内部有多个 task 调用和 interrupt），这保证了中间结果能及时流式输出。

## identifier() 函数：函数标识

```python
# libs/langgraph/langgraph/pregel/_call.py
def identifier(obj: Any, name: str | None = None) -> str | None:
    """Return the module and name of an object."""
    if isinstance(obj, PregelNode):
        obj = obj.bound
    if isinstance(obj, RunnableSeq):
        obj = obj.steps[0]
    if isinstance(obj, RunnableCallable):
        obj = obj.func
    if name is None:
        name = getattr(obj, "__qualname__", None)
    if name is None:
        name = getattr(obj, "__name__", None)
    if name is None:
        return None

    module_name = getattr(obj, "__module__", None)
    ...
```

`identifier()` 会穿透各种包装层找到原始函数，获取其 `__qualname__` 和 `__module__`。这个标识符用于：
- 缓存 key 生成（`CACHE_NS_WRITES` + identifier）
- 调试和日志输出
- 函数去重（CACHE 字典的 key）

## _TaskFunction 的 cache 操作

```python
# libs/langgraph/langgraph/func/__init__.py
def clear_cache(self, cache: BaseCache) -> None:
    """Clear the cache for this task."""
    if self.cache_policy is not None:
        cache.clear(
            ((CACHE_NS_WRITES, identifier(self.func) or "__dynamic__"),)
        )

async def aclear_cache(self, cache: BaseCache) -> None:
    """Clear the cache for this task."""
    if self.cache_policy is not None:
        await cache.aclear(
            ((CACHE_NS_WRITES, identifier(self.func) or "__dynamic__"),)
        )
```

缓存的 namespace 是 `(CACHE_NS_WRITES, func_identifier)`。如果函数没有可识别的标识符（如 lambda 或动态创建的函数），使用 `"__dynamic__"` 作为后备。

## 与 StateGraph 的对比总结

| 维度 | StateGraph | Functional API |
|------|-----------|----------------|
| 编程模型 | 声明式（图结构） | 命令式（函数调用） |
| 底层结构 | 多节点 Pregel 图 | 单节点 Pregel 图 |
| Channel 数量 | N 个（每个状态字段一个） | 3 个（START/END/PREVIOUS） |
| 状态管理 | 显式 State schema + reducer | `previous` 参数 + `entrypoint.final` |
| 路由 | `add_edge` / `add_conditional_edges` | 函数内 if/else 逻辑 |
| 并发 | 同一 superstep 的多节点 | `asyncio.gather` / futures |
| 子图支持 | 原生支持 | 可作为 StateGraph 的节点 |
| 可视化 | 丰富的拓扑图 | 简单（单节点） |

## 完整实战示例

```python
import time
import asyncio
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

@task
def compose_essay(topic: str) -> str:
    time.sleep(1.0)
    return f"An essay about {topic}"

@task
def check_grammar(text: str) -> str:
    time.sleep(0.5)
    return f"Grammar check: {text[:50]}... OK"

@entrypoint(checkpointer=InMemorySaver())
def review_workflow(topic: str) -> dict:
    """Essay generation and review workflow."""
    # 并行执行 essay 生成和语法检查准备
    essay_future = compose_essay(topic)
    essay = essay_future.result()

    grammar_future = check_grammar(essay)
    grammar = grammar_future.result()

    # interrupt 等待人工审核
    human_review = interrupt({
        "question": "Please provide a review",
        "essay": essay,
        "grammar": grammar
    })

    return {
        "essay": essay,
        "grammar": grammar,
        "review": human_review,
    }

config = {"configurable": {"thread_id": "review-1"}}

# 第一次调用：生成 essay 后中断
for result in review_workflow.stream("cats", config):
    print(result)

# 恢复执行：compose_essay 和 check_grammar 不会重新执行
for result in review_workflow.stream(
    Command(resume="Great essay!"),
    config
):
    print(result)
```

## Functional API 的限制

1. **单入口**：每个 entrypoint 只有一个入口函数
2. **不支持生成器**：不能使用 yield，需要用 StreamWriter
3. **单节点图**：底层只有一个节点，所有逻辑在一个 superstep 内
4. **可视化有限**：图结构简单，不如 StateGraph 的拓扑图直观
5. **子图限制**：entrypoint 可以作为 StateGraph 的节点，但在 entrypoint 内部嵌入子图需要额外处理

## 本章要点

1. **Functional API 是 Pregel 的语法糖**：`@entrypoint` 创建一个单节点的 Pregel 图，包含 START、END、PREVIOUS 三个 channel。底层执行引擎与 StateGraph 完全相同。

2. **@task 返回 Future**：调用 task 不会立即执行，而是返回 `SyncAsyncFuture`。通过 `.result()` 或 `await` 获取结果。这种设计天然支持并发。

3. **entrypoint.final 分离返回值与持久化值**：`value` 返回给调用者，`save` 写入 checkpoint。下次执行时通过 `previous` 参数访问。

4. **函数签名约束**：entrypoint 函数必须有至少一个参数（作为输入）。`previous`、`config`、`runtime` 是可选的注入参数。不支持 generator 函数。

5. **task 的 Runnable 包装**：同步函数自动创建异步版本。task 执行结果写入 RETURN channel，被 checkpointer 记录，支持恢复时跳过已完成的 task。

6. **cache 与 identifier**：task 的缓存 key 基于函数的模块路径和限定名（`module.qualname`）。动态创建的函数使用 `"__dynamic__"` 作为后备标识。

7. **类型推导的三种情况**：`entrypoint.final[R, S]` 分别提取 output 和 save 类型；未参数化的 `entrypoint.final` 使用 Any；普通类型两者相同。

8. **stream_eager=True**：Functional API 的图急切发射 stream 事件，确保长时间运行的 entrypoint 中间结果能及时输出。

9. **严格序列化白名单**：在 `STRICT_MSGPACK_ENABLED` 模式下，基于函数签名类型注解构建白名单，确保 checkpoint 安全。

10. **与 StateGraph 互补**：Functional API 适合函数式工作流，StateGraph 适合图结构化工作流。两者底层共享同一个 Pregel 引擎。
