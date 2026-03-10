# 第 9 章 Streaming：六种输出模式

在 LLM 应用中，流式输出不仅仅是 "体验优化"，更是实现 human-in-the-loop、实时调试、token 级监控等高级功能的基础设施。LangGraph 的 streaming 系统远比一般框架复杂——它支持六种 stream_mode，并且在同步与异步路径上采用了完全不同的背压与缓冲策略。

本章将从类型定义出发，逐层剖析 streaming 的完整管线：从 `StreamMode` 枚举到 `StreamPart` 类型系统，从 `SyncQueue` / `AsyncQueue` 的缓冲机制到 `StreamMessagesHandler` 的 token 级拦截，最后讨论 eager mode 与 `_output()` 函数中的 v1/v2 协议差异。

## StreamMode 枚举：七个字符串的世界

LangGraph 用一个 `Literal` 类型定义了所有合法的 stream mode：

```python
# langgraph/types.py
StreamMode = Literal[
    "values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"
]
```

每种模式有截然不同的语义：

| 模式 | 触发时机 | 数据内容 |
|---|---|---|
| `values` | 每个 step 结束后 | 完整的 state（所有 output channel 的当前值） |
| `updates` | 每个 node 执行完毕后 | 仅该 node 返回的增量更新 |
| `messages` | LLM 产生每个 token 时 | `(AIMessageChunk, metadata)` 二元组 |
| `custom` | 节点内部调用 `StreamWriter` 时 | 用户自定义的任意数据 |
| `checkpoints` | checkpoint 创建时 | 等同于 `get_state()` 的返回格式 |
| `tasks` | task 开始和结束时 | `TaskPayload` 或 `TaskResultPayload` |
| `debug` | 同 checkpoints + tasks | 包含 step、timestamp、type 的调试信封 |

这些模式可以任意组合——在 `stream()` / `astream()` 中传入一个列表即可同时启用多种模式：

```python
async for chunk in graph.astream(input, stream_mode=["updates", "messages", "custom"]):
    ...
```

当 `stream_mode` 是列表时，v1 协议下输出格式变为 `(mode, data)` 元组；v2 协议下则统一为 `StreamPart` TypedDict。

## StreamPart 类型系统

LangGraph v2 引入了一套完整的 TypedDict 来表示每种 stream 输出。每个 StreamPart 都包含 `type`、`ns`（namespace）和 `data` 三个字段，可以通过 `type` 字段进行类型窄化：

```python
# langgraph/types.py
class ValuesStreamPart(TypedDict, Generic[OutputT]):
    type: Literal["values"]
    ns: tuple[str, ...]
    data: OutputT
    interrupts: tuple[Interrupt, ...]

class UpdatesStreamPart(TypedDict):
    type: Literal["updates"]
    ns: tuple[str, ...]
    data: dict[str, Any]

class MessagesStreamPart(TypedDict):
    type: Literal["messages"]
    ns: tuple[str, ...]
    data: tuple[AnyMessage, dict[str, Any]]

class CustomStreamPart(TypedDict):
    type: Literal["custom"]
    ns: tuple[str, ...]
    data: Any

class CheckpointStreamPart(TypedDict, Generic[StateT]):
    type: Literal["checkpoints"]
    ns: tuple[str, ...]
    data: CheckpointPayload[StateT]

class TasksStreamPart(TypedDict):
    type: Literal["tasks"]
    ns: tuple[str, ...]
    data: TaskPayload | TaskResultPayload

class DebugStreamPart(TypedDict, Generic[StateT]):
    type: Literal["debug"]
    ns: tuple[str, ...]
    data: DebugPayload[StateT]
```

最终，所有这些类型通过 `TypeAliasType` 合并为一个 discriminated union：

```python
# langgraph/types.py
StreamPart = TypeAliasType(
    "StreamPart",
    ValuesStreamPart[OutputT]
    | UpdatesStreamPart
    | MessagesStreamPart
    | CustomStreamPart
    | CheckpointStreamPart[StateT]
    | TasksStreamPart
    | DebugStreamPart[StateT],
    type_params=(OutputT, StateT),
)
```

使用时只需判断 `part["type"]` 即可获得完整的类型推导：

```python
async for part in graph.astream(input, version="v2"):
    if part["type"] == "values":
        part["data"]  # OutputT
    elif part["type"] == "messages":
        part["data"]  # tuple[BaseMessage, dict]
    elif part["type"] == "custom":
        part["data"]  # Any
```

`ns` 字段表示 namespace——一个字符串元组，标记了当前输出来自哪个（子）图的哪个节点。根图的 `ns` 是空元组 `()`，子图的格式类似 `("parent_node:<task_id>", "child_node:<task_id>")`。

## StreamProtocol 与 StreamChunk

在引擎内部，所有 stream 数据都以三元组 `(namespace, mode, payload)` 的形式流转，这个类型被定义为 `StreamChunk`：

```python
# langgraph/pregel/protocol.py
StreamChunk = tuple[tuple[str, ...], str, Any]
```

`StreamProtocol` 是一个极简的包装器，将一个 callable 和一组允许的 mode 绑定在一起：

```python
# langgraph/pregel/protocol.py
class StreamProtocol:
    __slots__ = ("modes", "__call__")

    modes: set[StreamMode]
    __call__: Callable[[Self, StreamChunk], None]

    def __init__(
        self,
        __call__: Callable[[StreamChunk], None],
        modes: set[StreamMode],
    ) -> None:
        self.__call__ = cast(Callable[[Self, StreamChunk], None], __call__)
        self.modes = modes
```

在 `PregelLoop` 内部，`output_writes()` 方法通过检查 `self.stream.modes` 来决定是否 emit 某种类型的事件。例如，只有当 `"updates"` 在 `stream_modes` 中时，才会调用 `map_output_updates` 生成 updates 事件：

```python
# langgraph/pregel/_loop.py (output_writes 方法)
def output_writes(
    self, task_id: str, writes: WritesT, *, cached: bool = False
) -> None:
    if task := self.tasks.get(task_id):
        # ... 省略 hidden 检查 ...
        if writes[0][0] == INTERRUPT:
            # 中断处理...
            stream_modes = self.stream.modes if self.stream else []
            if "updates" in stream_modes:
                self._emit("updates", lambda: iter(interrupts))
            if "values" in stream_modes:
                current_values = read_channels(self.channels, self.output_keys)
                if isinstance(current_values, dict):
                    current_values[INTERRUPT] = interrupts[0][INTERRUPT]
                    self._emit("values", lambda: iter([current_values]))
        elif writes[0][0] != ERROR:
            self._emit(
                "updates",
                map_output_updates,
                self.output_keys,
                [(task, writes)],
                cached,
            )
        if not cached:
            self._emit(
                "tasks",
                map_debug_task_results,
                (task, writes),
                self.stream_keys,
            )
```

注意 `tasks` 模式的特殊性——它**不区分是否在 stream_modes 中**，因为 `_emit` 内部会再做一次 mode 检查。而 `cached` 标志决定了是否发送 task result 事件：缓存命中的 write 不会触发 tasks 事件。

## 背压与缓冲：SyncQueue 与 AsyncQueue

stream 方法需要在 "图执行" 和 "数据消费" 之间建立一个缓冲通道。LangGraph 为同步和异步路径分别实现了两套队列。

### SyncQueue：基于 Semaphore 的无界队列

```python
# langgraph/_internal/_queue.py
class SyncQueue:
    def __init__(self):
        self._queue = deque()
        self._count = Semaphore(0)

    def put(self, item, block=True, timeout=None):
        self._queue.append(item)
        self._count.release()

    def get(self, block=False, timeout=None):
        if not self._count.acquire(block, timeout):
            raise queue.Empty
        try:
            return self._queue.popleft()
        except IndexError:
            raise queue.Empty

    def wait(self, block=True, timeout=None):
        self._count.wait(block, timeout)
```

`SyncQueue` 不继承自标准库的 `queue.Queue`，而是基于 `deque` + 自定义 `Semaphore` 从头构建。关键设计：

1. **`put` 永不阻塞**——无界队列，数据直接 append 到 deque，然后 release semaphore
2. **`get` 默认非阻塞**——`block=False` 意味着在 stream 循环中采用 "poll" 模式
3. **`wait` 方法**——这是标准 `queue.Queue` 没有的。它阻塞直到有数据可用，但**不消费数据**。这是实现 eager mode 的关键

自定义的 `Semaphore` 同样扩展了标准库，增加了 `wait` 方法——利用条件变量（`self._cond`）阻塞直到 semaphore 值非零，但不 acquire。

### AsyncQueue：基于 asyncio.Queue 的扩展

`AsyncQueue` 直接继承 `asyncio.Queue`，同样只增加了一个 `wait()` 方法。实现逻辑参考了 `asyncio.Queue.get()` 的源码，但移除了实际消费数据的步骤——它利用 `_getters` 内部列表注册一个 Future，等到 `put_nowait()` 被调用时自动唤醒。

## stream() 的完整流程

下面以同步的 `stream()` 为例，梳理完整的数据流转：

```python
# langgraph/pregel/main.py (stream 方法核心逻辑)
def stream(self, input, config=None, *, stream_mode=None, ...):
    # 1. 确定 stream_mode
    if stream_mode is None:
        stream_mode = (
            "values"
            if config is not None and CONFIG_KEY_TASK_ID in config.get(CONF, {})
            else self.stream_mode
        )

    # 2. 创建队列
    stream = SyncQueue()

    # 3. 计算 stream_modes 集合（合并 print_mode）
    stream_modes, ... = self._defaults(config, stream_mode=stream_mode, ...)

    # 4. 构建 StreamProtocol
    # StreamProtocol(stream.put, stream_modes)

    # 5. 进入主循环
    with SyncPregelLoop(
        input,
        stream=StreamProtocol(stream.put, stream_modes),
        ...
    ) as loop:
        while loop.tick():
            for _ in runner.tick(..., get_waiter=get_waiter):
                yield from _output(stream_mode, ..., stream.get, queue.Empty, ...)
```

关键步骤：

1. **stream_mode 默认值推导**：如果作为子图节点运行（`CONFIG_KEY_TASK_ID` 存在），默认使用 `"values"` 模式
2. **_defaults 合并**：将 `stream_mode` 和 `print_mode` 合并为 `stream_modes` 集合
3. **StreamProtocol 创建**：将 `stream.put` 和 `stream_modes` 绑定，传入 loop
4. **tick 循环中 yield**：每次 runner.tick() yield 控制权时，从队列中取出所有可用数据

### astream() 的异步差异

`astream()` 与 `stream()` 的结构高度相似，但有几个关键差异：

```python
# langgraph/pregel/main.py (astream 方法)
async def astream(self, input, config=None, *, stream_mode=None, ...):
    stream = AsyncQueue()
    aioloop = asyncio.get_running_loop()
    stream_put = cast(
        Callable[[StreamChunk], None],
        partial(aioloop.call_soon_threadsafe, stream.put_nowait),
    )
```

**差异一：线程安全写入**。`astream` 使用 `aioloop.call_soon_threadsafe(stream.put_nowait)` 而非直接调用 `stream.put_nowait`。这是因为 `messages` 和 `custom` 模式下的回调可能在非 asyncio 线程中触发（例如 LangChain 的 callback handler），需要通过 `call_soon_threadsafe` 将写入操作调度回事件循环线程。

**差异二：do_stream 检测**。`astream` 会检查是否存在 `_StreamingCallbackHandler`，如果存在则在 runner 中使用 `astream` 而非 `ainvoke` 执行 task：

```python
# langgraph/pregel/main.py
do_stream = (
    next(
        (
            True
            for h in run_manager.handlers
            if isinstance(h, _StreamingCallbackHandler)
            and not isinstance(h, StreamMessagesHandler)
        ),
        False,
    )
    if _StreamingCallbackHandler is not None
    else False
)
```

**差异三：waiter 清理**。异步版本需要额外的 `_cleanup_waiter` 来处理取消场景，避免 asyncio Task 泄漏：

```python
# langgraph/pregel/main.py
async def _cleanup_waiter() -> None:
    nonlocal waiter
    with contextlib.suppress(Exception):
        if hasattr(stream, "_count"):
            stream._count.release()
    t = waiter
    waiter = None
    if t is not None and not t.done():
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t
```

## eager_mode 与 messages/custom 的特殊处理

`stream_eager` 是 `Pregel` 类上的一个属性，默认为 `False`。但当 `stream_mode` 包含 `"messages"` 或 `"custom"` 时，框架自动启用 eager 行为：

```python
# langgraph/pregel/main.py (stream 方法内)
get_waiter: Callable[[], concurrent.futures.Future[None]] | None = None
if (
    self.stream_eager
    or subgraphs
    or "messages" in stream_modes
    or "custom" in stream_modes
):
    waiter: concurrent.futures.Future | None = None
    loop.stack.callback(stream._count.release)

    def get_waiter() -> concurrent.futures.Future[None]:
        nonlocal waiter
        if waiter is None or waiter.done():
            waiter = loop.submit(stream.wait)
            return waiter
        else:
            return waiter
```

**非 eager 模式**下，`get_waiter` 为 `None`，runner 只在 step 间隙 yield 控制权，此时从队列中批量取出所有 stream 数据。这意味着 `values` 和 `updates` 模式的输出天然与 step 边界对齐。

**eager 模式**下，`get_waiter` 返回一个 Future，该 Future 在队列中有新数据时完成。这使得 runner 可以在 node 执行过程中就 yield 控制权，让上层 generator 立即消费新到达的 stream 数据。对于 `messages` 模式（token 级流式），这是必须的——否则用户要等到整个 step 结束才能看到 LLM 的输出。

同步版本在退出时通过 `loop.stack.callback(stream._count.release)` 释放 semaphore，确保阻塞在 `wait()` 上的 waiter 能被唤醒并正常退出。

## messages 模式的回调拦截

`messages` 模式的实现与其他模式完全不同——它不是在 loop 层面发射事件，而是通过 LangChain 的 callback 系统拦截 LLM 的 token 输出。

```python
# langgraph/pregel/main.py
if "messages" in stream_modes:
    ns_ = cast(str | None, config[CONF].get(CONFIG_KEY_CHECKPOINT_NS))
    run_manager.inheritable_handlers.append(
        StreamMessagesHandler(
            stream.put,
            subgraphs,
            parent_ns=tuple(ns_.split(NS_SEP)) if ns_ else None,
        )
    )
```

`StreamMessagesHandler` 继承自 `BaseCallbackHandler` 和 `_StreamingCallbackHandler`，并实现了几个关键回调：

```python
# langgraph/pregel/_messages.py
class StreamMessagesHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    run_inline = True  # 在主线程中运行，避免顺序/锁问题

    def on_chat_model_start(self, serialized, messages, *, run_id, tags=None,
                            metadata=None, **kwargs):
        if metadata and (not tags or (TAG_NOSTREAM not in tags)):
            ns = tuple(cast(str, metadata["langgraph_checkpoint_ns"]).split(NS_SEP))[:-1]
            if not self.subgraphs and len(ns) > 0 and ns != self.parent_ns:
                return
            self.metadata[run_id] = (ns, metadata)

    def on_llm_new_token(self, token, *, chunk=None, run_id, **kwargs):
        if not isinstance(chunk, ChatGenerationChunk):
            return
        if meta := self.metadata.get(run_id):
            self._emit(meta, chunk.message)

    def on_chain_end(self, response, *, run_id, **kwargs):
        if meta := self.metadata.pop(run_id, None):
            if isinstance(response, Command):
                self._find_and_emit_messages(meta, response.update)
            else:
                self._find_and_emit_messages(meta, response)
```

设计要点：

1. **`run_inline = True`**：确保回调在主线程中执行，避免多线程竞争
2. **`TAG_NOSTREAM` 过滤**：带有 `TAG_NOSTREAM` 标签的 LLM 调用不会产生 messages 事件
3. **去重机制**：通过 `self.seen` 集合跟踪已发射的 message ID，防止在 `on_chain_end` 中重复发射 `on_llm_new_token` 已经发过的消息
4. **namespace 过滤**：非 subgraphs 模式下，只发射根图的消息（`len(ns) > 0 and ns != self.parent_ns` 时跳过）

## custom 模式与 StreamWriter

`custom` 模式通过 `StreamWriter` 实现——一个简单的 `Callable[[Any], None]`：

```python
# langgraph/types.py
StreamWriter = Callable[[Any], None]
```

在 `stream()` 方法中，当 `"custom"` 在 stream_modes 中时，框架创建一个 closure 作为 stream_writer：

```python
# langgraph/pregel/main.py (stream 方法)
if "custom" in stream_modes:
    def stream_writer(c: Any) -> None:
        stream.put(
            (
                tuple(
                    get_config()[CONF][CONFIG_KEY_CHECKPOINT_NS].split(NS_SEP)[:-1]
                ),
                "custom",
                c,
            )
        )
```

这个 closure 在节点函数中通过依赖注入获取。当不使用 custom 模式时，stream_writer 退化为空函数 `lambda c: None`。

## _output 函数：v1 与 v2 的协议转换

所有 stream 数据最终通过 `_output()` 函数从队列中取出并 yield 给用户。这个函数同时处理 v1 和 v2 两种协议：

```python
# langgraph/pregel/main.py
def _output(
    stream_mode: StreamMode | Sequence[StreamMode],
    print_mode: StreamMode | Sequence[StreamMode],
    stream_subgraphs: bool,
    getter: Callable[[], tuple[tuple[str, ...], str, Any]],
    empty_exc: type[Exception],
    version: Literal["v1", "v2"] = "v1",
    output_mapper: Callable[[Any], Any] | None = None,
    state_mapper: Callable[[Any], Any] | None = None,
) -> Iterator:
    while True:
        try:
            ns, mode, payload = getter()
        except empty_exc:
            break
        if mode in print_mode:
            print(...)  # 调试打印
        if mode in stream_mode:
            if version == "v2":
                if mode == "values":
                    ints: tuple[Interrupt, ...] = ()
                    if isinstance(payload, dict):
                        ints = payload.pop(INTERRUPT, ())
                        if output_mapper:
                            payload = output_mapper(payload)
                    yield {"type": mode, "ns": ns, "data": payload, "interrupts": ints}
                elif mode in ("checkpoints", "debug"):
                    if state_mapper:
                        _coerce_checkpoint_values(payload, state_mapper)
                    yield {"type": mode, "ns": ns, "data": payload}
                else:
                    yield {"type": mode, "ns": ns, "data": payload}
            elif stream_subgraphs and isinstance(stream_mode, list):
                yield (ns, mode, payload)
            elif isinstance(stream_mode, list):
                yield (mode, payload)
            elif stream_subgraphs:
                yield (ns, payload)
            else:
                yield payload
```

v2 协议的关键改进：

1. **统一的 TypedDict 格式**：所有输出都是 `{"type": ..., "ns": ..., "data": ...}` 形式
2. **interrupts 提取**：`values` 模式下，`__interrupt__` 从 payload dict 中 pop 出来，放入独立的 `interrupts` 字段
3. **类型转换**：通过 `output_mapper` 和 `state_mapper` 将内部 dict 转换为用户定义的 Pydantic model 或 dataclass

v1 协议则根据 `stream_mode` 是否为列表、是否启用 subgraphs 来决定 yield 元组的层级。

## print_mode：调试利器

`stream()` 和 `astream()` 都接受一个 `print_mode` 参数，语义与 `stream_mode` 相同，但效果完全不同——它只将数据打印到控制台，不影响 yield 的输出。`print_mode` 的 mode 会被合并到 `stream_modes` 集合中，确保 loop 层面会生成对应的事件：

```python
# langgraph/pregel/main.py (_defaults 方法)
if isinstance(stream_mode, str):
    stream_modes = {stream_mode}
else:
    stream_modes = set(stream_mode)
if isinstance(print_mode, str):
    stream_modes.add(print_mode)
else:
    stream_modes.update(print_mode)
```

当 `debug=True` 时，框架自动将 `print_mode` 设置为 `["updates", "values"]`，在控制台输出每一步的详细信息。

## 本章要点

1. **六种 stream_mode** 覆盖了从全量 state 到 token 级粒度的所有需求；`debug` 模式是 `checkpoints` + `tasks` 的超集，用于开发调试
2. **StreamPart 类型系统**（v2）通过 discriminated union 提供完整的类型安全；v1 协议通过元组层级区分单模式、多模式、subgraphs 等场景
3. **SyncQueue 与 AsyncQueue** 都扩展了标准库，增加了 `wait()` 方法以支持 eager 模式下的非消费式等待
4. **eager_mode** 是 messages 和 custom 模式正常工作的前提——它通过 `get_waiter` 机制让 runner 在 node 执行中途就 yield 控制权
5. **messages 模式**不走 loop 层的 `output_writes`，而是通过 `StreamMessagesHandler` 回调直接注入 stream 队列，实现 token 级延迟
6. **`_output()` 函数**是所有模式的统一出口，负责从队列中取数据、过滤 mode、转换协议版本，并处理 `print_mode` 的调试输出
7. **StreamProtocol** 将 `put` 回调与 `modes` 集合绑定，使得 loop 层可以按需决定是否发射特定模式的事件，避免不必要的计算开销
