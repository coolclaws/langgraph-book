# 第 9 章 Streaming：六种输出模式

LangGraph 的流式输出系统是连接运行时引擎与用户体验的桥梁。当 LLM 逐 token 生成、
节点逐步完成时，用户不必等待整个图执行完毕即可看到中间结果。本章深入解析
`stream_mode` 的六种模式、`StreamPart` 类型系统、`StreamMessagesHandler` 回调机制
以及背压与缓冲的工程实现。

> 源码路径
> - `libs/langgraph/langgraph/types.py` -- StreamMode、StreamPart 类型定义
> - `libs/langgraph/langgraph/pregel/main.py` -- stream/astream 方法
> - `libs/langgraph/langgraph/pregel/_messages.py` -- messages 模式的回调处理
> - `libs/langgraph/langgraph/pregel/_loop.py` -- _emit、output_writes
> - `libs/langgraph/langgraph/pregel/protocol.py` -- StreamProtocol

---

## 9.1 StreamMode 枚举

```python
# libs/langgraph/langgraph/types.py

StreamMode = Literal[
    "values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"
]
```

七个字面量值（其中 `checkpoints` 和 `tasks` 也可以通过 `debug` 模式一起获取），
每个都对应不同的信息粒度和使用场景：

| 模式 | 触发时机 | 数据内容 | 典型场景 |
|---|---|---|---|
| `values` | 每个超步结束 | 完整状态快照 | 跟踪全局状态变化 |
| `updates` | 每个任务完成 | 节点名 + 输出值 | UI 显示节点级进度 |
| `messages` | LLM 每个 token | (AIMessageChunk, metadata) | 实时打字机效果 |
| `custom` | 节点内主动调用 StreamWriter | 任意数据 | 自定义进度报告 |
| `checkpoints` | checkpoint 创建时 | CheckpointPayload | 调试、状态回溯 |
| `tasks` | 任务开始/完成 | TaskPayload / TaskResultPayload | 监控执行进度 |
| `debug` | 合并 checkpoints + tasks | 带 step/timestamp 的包装 | 全方位调试 |

---

## 9.2 StreamPart 类型系统

LangGraph 为每种流式模式定义了精确的 TypedDict 类型，组成一个 discriminated union：

```python
# libs/langgraph/langgraph/types.py

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

### 9.2.1 StreamPart union

```python
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

这个 discriminated union 通过 `part["type"]` 字段区分，配合 v2 streaming API 使用：

```python
async for part in graph.astream(input, version="v2"):
    if part["type"] == "values":
        part["data"]  # OutputT -- 完整状态
    elif part["type"] == "messages":
        part["data"]  # tuple[BaseMessage, dict] -- (消息, 元数据)
    elif part["type"] == "custom":
        part["data"]  # Any -- 用户自定义
```

### 9.2.2 所有 StreamPart 的公共字段

每个 StreamPart 都包含两个公共字段：

- `type` -- 模式标识符，用于类型窄化
- `ns` -- namespace 元组，标识事件来源的图层级

对于顶层图，`ns` 为空元组 `()`。对于子图，`ns` 包含路径信息，
例如 `("parent_node:task_id", "child_node:task_id")`。

---

## 9.3 Payload 类型详解

### 9.3.1 TaskPayload（任务开始）

```python
# libs/langgraph/langgraph/types.py

class TaskPayload(TypedDict):
    id: str           # 任务唯一 ID
    name: str         # 节点名
    input: Any        # 输入数据
    triggers: list[str]  # 触发 channel 列表
```

### 9.3.2 TaskResultPayload（任务完成）

```python
class TaskResultPayload(TypedDict):
    id: str           # 任务唯一 ID
    name: str         # 节点名
    error: str | None       # 错误信息（失败时）
    interrupts: list[dict]  # 中断信息
    result: dict[str, Any]  # channel 写入结果
```

### 9.3.3 CheckpointPayload

```python
class CheckpointPayload(TypedDict, Generic[StateT]):
    config: RunnableConfig | None
    metadata: CheckpointMetadata
    values: StateT         # 当前状态值
    next: list[str]        # 下一步要执行的节点
    parent_config: RunnableConfig | None
    tasks: list[CheckpointTask]
```

### 9.3.4 DebugPayload

```python
DebugPayload = TypeAliasType(
    "DebugPayload",
    _DebugCheckpointPayload[StateT]
    | _DebugTaskPayload
    | _DebugTaskResultPayload,
    type_params=(StateT,),
)
```

Debug 事件在 checkpoint 和 task 数据基础上添加了 `step`、`timestamp` 和 `type` 字段：

```python
class _DebugTaskPayload(TypedDict):
    step: int               # 超步编号
    timestamp: str          # ISO 8601 时间戳
    type: Literal["task"]   # 事件类型
    payload: TaskPayload
```

---

## 9.4 StreamProtocol：流式管道

```python
# libs/langgraph/langgraph/pregel/protocol.py

StreamChunk = tuple[tuple[str, ...], str, Any]
# (namespace, mode, data)

class StreamProtocol:
    __call__: Callable[[StreamChunk], None]
    modes: set[str]
```

`StreamProtocol` 是一个简单的协议：一个可调用对象加上它支持的模式集合。
`StreamChunk` 是内部传输格式 -- `(namespace, mode, data)` 三元组。

在 `stream()` 方法中创建：

```python
# libs/langgraph/langgraph/pregel/main.py  stream() 内

stream = SyncQueue()
# ...
with SyncPregelLoop(
    input,
    stream=StreamProtocol(stream.put, stream_modes),
    ...
) as loop:
```

`stream.put` 是一个 `SyncQueue`（封装 `queue.Queue`）的 put 方法。
所有流式事件最终都通过 `stream.put((ns, mode, data))` 进入队列。

### 9.4.1 DuplexStream

当子图需要将事件传递给父图时，使用 `DuplexStream`：

```python
# libs/langgraph/langgraph/pregel/_loop.py

def DuplexStream(*streams: StreamProtocol) -> StreamProtocol:
    def __call__(value: StreamChunk) -> None:
        for stream in streams:
            if value[1] in stream.modes:
                stream(value)

    return StreamProtocol(
        __call__,
        {mode for s in streams for mode in s.modes}
    )
```

每个事件被发送到所有 stream，但只有支持该 mode 的 stream 才会实际处理。

---

## 9.5 stream() 方法：流式输出的完整流程

```python
# libs/langgraph/langgraph/pregel/main.py

def stream(
    self,
    input: InputT | Command | None,
    config: RunnableConfig | None = None,
    *,
    stream_mode: StreamMode | Sequence[StreamMode] | None = None,
    print_mode: StreamMode | Sequence[StreamMode] = (),
    output_keys: str | Sequence[str] | None = None,
    interrupt_before: All | Sequence[str] | None = None,
    interrupt_after: All | Sequence[str] | None = None,
    durability: Durability | None = None,
    subgraphs: bool = False,
    debug: bool | None = None,
    version: Literal["v1", "v2"] = "v1",
    **kwargs,
) -> Iterator[dict[str, Any] | Any]:
```

### 9.5.1 模式初始化

```python
if stream_mode is None:
    stream_mode = (
        "values"
        if config is not None and CONFIG_KEY_TASK_ID in config.get(CONF, {})
        else self.stream_mode
    )
if debug or self.debug:
    print_mode = ["updates", "values"]
```

当作为子图节点调用时（`CONFIG_KEY_TASK_ID` 存在），默认切换到 `"values"` 模式。
启用 debug 时，自动添加 print_mode 来显示 updates 和 values。

### 9.5.2 messages 模式的设置

```python
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

messages 模式通过 LangChain 的回调系统实现。`StreamMessagesHandler` 被注册为
`inheritable_handler`，这意味着它会被传递给所有子 Runnable（包括 LLM 调用）。

### 9.5.3 custom 模式的设置

```python
if "custom" in stream_modes:
    def stream_writer(c: Any) -> None:
        stream.put((
            tuple(
                get_config()[CONF][CONFIG_KEY_CHECKPOINT_NS].split(NS_SEP)[:-1]
            ),
            "custom",
            c,
        ))
```

custom 模式通过 `stream_writer` 函数实现。它被注入到 `Runtime` 对象中，
节点内部通过 `StreamWriter` 参数接收并调用。

### 9.5.4 并发流式输出

```python
# libs/langgraph/langgraph/pregel/main.py  stream() 内

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

对于需要实时流式输出的模式（messages、custom、subgraphs），设置了 `get_waiter`。
它通过 `stream.wait` 在后台等待新的流式数据，当数据到达时唤醒主循环产出输出。

`stream._count` 是一个信号量（`threading.Semaphore`）。`stream.wait` 阻塞等待
信号量，`stream.put` 在放入数据后释放信号量。退出时通过
`loop.stack.callback(stream._count.release)` 确保 waiter 不会永久阻塞。

### 9.5.5 主循环中的输出产出

```python
while loop.tick():
    for task in loop.match_cached_writes():
        loop.output_writes(task.id, task.writes, cached=True)
    for _ in runner.tick(
        [t for t in loop.tasks.values() if not t.writes],
        timeout=self.step_timeout,
        get_waiter=get_waiter,
        schedule_task=loop.accept_push,
    ):
        # 每次 runner.tick yield 时，消费队列中的所有事件
        yield from _output(
            stream_mode, print_mode, subgraphs,
            stream.get, queue.Empty, version,
            _output_mapper, _state_mapper,
        )
    loop.after_tick()
    if durability_ == "sync":
        loop._put_checkpoint_fut.result()

# 循环结束后，消费剩余事件
yield from _output(
    stream_mode, print_mode, subgraphs,
    stream.get, queue.Empty, version,
    _output_mapper, _state_mapper,
)
```

`_output` 函数从队列中取出所有可用的 StreamChunk，根据 `version` 参数
格式化后 yield 给调用方。

---

## 9.6 _emit：Loop 内部的事件发射

所有流式事件的发射最终通过 `PregelLoop._emit` 方法：

```python
# libs/langgraph/langgraph/pregel/_loop.py

def _emit(
    self,
    mode: StreamMode,
    values: Callable[P, Iterator[Any]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    if self.stream is None:
        return
    debug_remap = mode in ("checkpoints", "tasks") and "debug" in self.stream.modes
    if mode not in self.stream.modes and not debug_remap:
        return
    for v in values(*args, **kwargs):
        if mode in self.stream.modes:
            self.stream((self.checkpoint_ns, mode, v))
        # "debug" 模式将 "checkpoints" 和 "tasks" 包装为 debug 事件
        if debug_remap:
            self.stream((
                self.checkpoint_ns,
                "debug",
                {
                    "step": self.step - 1 if mode == "checkpoints" else self.step,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "checkpoint" if mode == "checkpoints"
                            else "task_result" if "result" in v
                            else "task",
                    "payload": v,
                },
            ))
```

设计要点：

1. **懒计算**：`values` 参数是一个 callable，只有当 stream 需要该 mode 时才被调用
2. **debug 复用**：`debug` 模式不是独立生成事件，而是在 `checkpoints` 和 `tasks` 事件上包装
3. **namespace 传递**：每个事件携带 `self.checkpoint_ns`，支持多层子图

### 9.6.1 各阶段的 _emit 调用

| 调用位置 | mode | 内容 |
|---|---|---|
| `tick()` checkpoint 后 | `"checkpoints"` | 当前 checkpoint 状态 |
| `tick()` 任务准备后 | `"tasks"` | 任务开始信息 |
| `after_tick()` 写入后 | `"values"` | 完整状态快照 |
| `output_writes()` 正常完成 | `"updates"` | 节点更新 |
| `output_writes()` interrupt | `"updates"` / `"values"` | 中断信息 |
| `output_writes()` 非缓存 | `"tasks"` | 任务结果 |

---

## 9.7 output_writes：任务级事件发射

```python
# libs/langgraph/langgraph/pregel/_loop.py

def output_writes(self, task_id, writes, *, cached=False):
    if task := self.tasks.get(task_id):
        # 跳过隐藏任务
        if TAG_HIDDEN in task.config.get("tags", EMPTY_SEQ):
            return

        if writes[0][0] == INTERRUPT:
            # PUSH Call 的 interrupt 由父任务处理
            if task.path[0] == PUSH and task.path[-1] is True:
                return
            interrupts = [{
                INTERRUPT: tuple(
                    v
                    for w in writes if w[0] == INTERRUPT
                    for v in (w[1] if isinstance(w[1], Sequence) else (w[1],))
                )
            }]
            if "updates" in stream_modes:
                self._emit("updates", lambda: iter(interrupts))
            if "values" in stream_modes:
                current_values = read_channels(self.channels, self.output_keys)
                if isinstance(current_values, dict):
                    current_values[INTERRUPT] = interrupts[0][INTERRUPT]
                    self._emit("values", lambda: iter([current_values]))
                else:
                    self._emit("values", lambda: iter(interrupts))

        elif writes[0][0] != ERROR:
            self._emit(
                "updates", map_output_updates,
                self.output_keys, [(task, writes)], cached,
            )

        if not cached:
            self._emit(
                "tasks", map_debug_task_results,
                (task, writes), self.stream_keys,
            )
```

三种分支：

1. **INTERRUPT 写入** -- 发射到 `updates` 和 `values` 流，包含中断详情
2. **正常写入** -- 通过 `map_output_updates` 发射到 `updates` 流
3. **任务结果** -- 非缓存结果通过 `map_debug_task_results` 发射到 `tasks` 流

`TAG_HIDDEN` 的任务（如内部路由节点）不产生任何流式输出。

---

## 9.8 StreamMessagesHandler：messages 模式

messages 模式是最复杂的流式模式，它需要拦截所有 LLM 调用的 token 流。
实现方式是通过 LangChain 的 callback 系统。

```python
# libs/langgraph/langgraph/pregel/_messages.py

class StreamMessagesHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    """A callback handler that implements stream_mode=messages.

    Collects messages from:
    (1) chat model stream events; and
    (2) node outputs.
    """

    run_inline = True  # 在主线程运行，避免排序和锁问题

    def __init__(self, stream, subgraphs, *, parent_ns=None):
        self.stream = stream       # stream.put 函数
        self.subgraphs = subgraphs # 是否包含子图消息
        self.metadata: dict[UUID, Meta] = {}  # run_id -> (ns, metadata)
        self.seen: set[int | str] = set()     # 去重集合
        self.parent_ns = parent_ns
```

### 9.8.1 消息来源一：LLM Token 流

```python
# libs/langgraph/langgraph/pregel/_messages.py

def on_chat_model_start(self, serialized, messages, *, run_id, metadata, tags, **kwargs):
    if metadata and (not tags or TAG_NOSTREAM not in tags):
        ns = tuple(cast(str, metadata["langgraph_checkpoint_ns"]).split(NS_SEP))[:-1]
        if not self.subgraphs and len(ns) > 0 and ns != self.parent_ns:
            return  # 非子图模式下跳过子图消息
        if tags:
            if filtered_tags := [t for t in tags if not t.startswith("seq:step")]:
                metadata["tags"] = filtered_tags
        self.metadata[run_id] = (ns, metadata)

def on_llm_new_token(self, token, *, chunk, run_id, **kwargs):
    if not isinstance(chunk, ChatGenerationChunk):
        return
    if meta := self.metadata.get(run_id):
        self._emit(meta, chunk.message)
```

当 LLM 开始生成时，`on_chat_model_start` 记录 `run_id -> (namespace, metadata)` 映射。
每个新 token 到达时，`on_llm_new_token` 从 chunk 中提取 `AIMessageChunk` 并发射。

元数据中包含的关键信息：

```python
metadata = {
    "langgraph_step": step,           # 当前超步
    "langgraph_node": name,           # 所在节点名
    "langgraph_triggers": triggers,   # 触发 channel
    "langgraph_checkpoint_ns": ns,    # checkpoint namespace
    "tags": filtered_tags,            # 过滤后的标签
}
```

### 9.8.2 消息来源二：节点输出

```python
# libs/langgraph/langgraph/pregel/_messages.py

def on_chain_start(self, serialized, inputs, *, run_id, tags, metadata, **kwargs):
    if (
        metadata
        and kwargs.get("name") == metadata.get("langgraph_node")
        and (not tags or TAG_HIDDEN not in tags)
    ):
        ns = tuple(cast(str, metadata["langgraph_checkpoint_ns"]).split(NS_SEP))[:-1]
        if not self.subgraphs and len(ns) > 0:
            return
        self.metadata[run_id] = (ns, metadata)
        # 记录输入中已有的消息 ID，用于去重
        for value in _state_values(inputs):
            if isinstance(value, BaseMessage):
                if value.id is not None:
                    self.seen.add(value.id)
            elif isinstance(value, Sequence) and not isinstance(value, str):
                for item in value:
                    if isinstance(item, BaseMessage):
                        if item.id is not None:
                            self.seen.add(item.id)

def on_chain_end(self, response, *, run_id, **kwargs):
    if meta := self.metadata.pop(run_id, None):
        if isinstance(response, Command):
            self._find_and_emit_messages(meta, response.update)
        elif isinstance(response, Sequence) and any(
            isinstance(value, Command) for value in response
        ):
            for value in response:
                if isinstance(value, Command):
                    self._find_and_emit_messages(meta, value.update)
                else:
                    self._find_and_emit_messages(meta, value)
        else:
            self._find_and_emit_messages(meta, response)
```

`on_chain_start` 识别节点级 chain（通过 `name == langgraph_node`），记录输入消息的 ID。
`on_chain_end` 从节点输出中提取所有 `BaseMessage`，去重后发射。

对于 `Command` 类型的输出，消息从 `response.update` 中提取。

### 9.8.3 去重机制

```python
# libs/langgraph/langgraph/pregel/_messages.py

def _emit(self, meta, message, *, dedupe=False):
    if dedupe and message.id in self.seen:
        return
    else:
        if message.id is None:
            message.id = str(uuid4())
        self.seen.add(message.id)
        self.stream((meta[0], "messages", (message, meta[1])))
```

去重通过 `self.seen` 集合实现。LLM token 流不去重（`dedupe=False`），
节点输出去重（`dedupe=True`）。这是因为 LLM token 是增量的，而节点输出
可能包含之前已经流式发送过的完整消息。

### 9.8.4 消息提取

```python
# libs/langgraph/langgraph/pregel/_messages.py

def _find_and_emit_messages(self, meta, response):
    if isinstance(response, BaseMessage):
        self._emit(meta, response, dedupe=True)
    elif isinstance(response, Sequence):
        for value in response:
            if isinstance(value, BaseMessage):
                self._emit(meta, value, dedupe=True)
    else:
        for value in _state_values(response):
            if isinstance(value, BaseMessage):
                self._emit(meta, value, dedupe=True)
            elif isinstance(value, Sequence):
                for item in value:
                    if isinstance(item, BaseMessage):
                        self._emit(meta, item, dedupe=True)
```

`_state_values` 从 dict、Pydantic model 或 dataclass 中提取顶层字段值：

```python
# libs/langgraph/langgraph/pregel/_messages.py

def _state_values(obj):
    if isinstance(obj, dict):
        return list(obj.values())
    elif isinstance(obj, BaseModel):
        return [getattr(obj, k) for k in type(obj).model_fields]
    elif is_dataclass(obj) and not isinstance(obj, type):
        return [getattr(obj, f.name) for f in fields(obj)]
    return ()
```

### 9.8.5 子图消息过滤

```python
# on_chat_model_start 内部
if not self.subgraphs and len(ns) > 0 and ns != self.parent_ns:
    return
```

当 `subgraphs=False` 时，只有来自当前图层级的消息才会被发射。
但有一个特殊情况：如果消息来自 `parent_ns` 所在的层级（即用户在节点内
显式调用了子图的 `stream(stream_mode="messages")`），仍然允许通过。

### 9.8.6 LLM 结束时的完整消息

```python
# libs/langgraph/langgraph/pregel/_messages.py

def on_llm_end(self, response, *, run_id, **kwargs):
    if meta := self.metadata.get(run_id):
        if response.generations and response.generations[0]:
            gen = response.generations[0][0]
            if isinstance(gen, ChatGeneration):
                self._emit(meta, gen.message, dedupe=True)
    self.metadata.pop(run_id, None)
```

当 LLM 生成结束时，发射完整的 `AIMessage`（而非 `AIMessageChunk`）。
使用 `dedupe=True` 避免与已发送的 token 重复。

---

## 9.9 Custom 模式：StreamWriter

custom 模式通过 `StreamWriter` 函数实现，它被注入到节点的参数中：

```python
# libs/langgraph/langgraph/pregel/main.py  stream() 内

if "custom" in stream_modes:
    def stream_writer(c: Any) -> None:
        stream.put((
            tuple(
                get_config()[CONF][CONFIG_KEY_CHECKPOINT_NS].split(NS_SEP)[:-1]
            ),
            "custom",
            c,
        ))
```

`stream_writer` 被存储在 `Runtime` 对象中：

```python
runtime = Runtime(
    context=...,
    store=store,
    stream_writer=stream_writer,  # 注入 StreamWriter
    previous=None,
)
```

节点内部使用时，通过参数注入获取：

```python
from langgraph.types import StreamWriter

def my_node(state: State, writer: StreamWriter):
    writer("Processing step 1...")  # 发射 custom 事件
    result = do_something()
    writer({"progress": 50})        # 可以是任意数据
    return result
```

当 `stream_mode` 不包含 `"custom"` 时，`stream_writer` 是一个空操作：

```python
else:
    def stream_writer(c: Any) -> None:
        pass
```

这个设计让节点代码无需关心是否有人在监听 custom 流。

---

## 9.10 Values 模式：状态快照

`values` 模式在三个时机发射：

### 9.10.1 超步完成后

```python
# libs/langgraph/langgraph/pregel/_loop.py  after_tick() 内

if not self.updated_channels.isdisjoint(
    (self.output_keys,) if isinstance(self.output_keys, str)
    else self.output_keys
):
    self._emit(
        "values", map_output_values,
        self.output_keys, writes, self.channels
    )
```

只有当输出 channel 被更新时才发射。`map_output_values` 读取所有输出 channel
的当前值，组装为完整的状态快照。

### 9.10.2 Interrupt 时

```python
# libs/langgraph/langgraph/pregel/_loop.py  output_writes() 内

if "values" in stream_modes:
    current_values = read_channels(self.channels, self.output_keys)
    if isinstance(current_values, dict):
        current_values[INTERRUPT] = interrupts[0][INTERRUPT]
        self._emit("values", lambda: iter([current_values]))
```

interrupt 时发射的 values 事件会附带 `__interrupt__` 键。

### 9.10.3 Resume 开始时

```python
# libs/langgraph/langgraph/pregel/_loop.py  _first() 内

if is_resuming:
    self._emit(
        "values", map_output_values, self.output_keys, True, self.channels
    )
```

resume 时立即发射当前状态，让客户端知道图从哪个状态继续。

---

## 9.11 Updates 模式：增量更新

```python
# libs/langgraph/langgraph/pregel/_loop.py  output_writes() 内

elif writes[0][0] != ERROR:
    self._emit(
        "updates", map_output_updates,
        self.output_keys, [(task, writes)], cached,
    )
```

`map_output_updates` 将任务的写入转换为 `{node_name: output}` 格式。
如果是缓存命中（`cached=True`），输出会被标记。

updates 模式是最常用的流式模式之一，因为它只包含增量变化，数据量远小于 values。
典型输出形如：

```python
{"agent": {"messages": [AIMessage(content="Hello")]}}
{"tool": {"messages": [ToolMessage(content="Result")]}}
```

---

## 9.12 Debug 模式：全面调试信息

debug 模式不是独立的事件流，而是 `checkpoints` 和 `tasks` 的聚合包装：

```python
# libs/langgraph/langgraph/pregel/_loop.py  _emit() 内

debug_remap = mode in ("checkpoints", "tasks") and "debug" in self.stream.modes
if debug_remap:
    self.stream((
        self.checkpoint_ns,
        "debug",
        {
            "step": self.step - 1 if mode == "checkpoints" else self.step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "checkpoint" if mode == "checkpoints"
                    else "task_result" if "result" in v
                    else "task",
            "payload": v,
        },
    ))
```

debug 事件的 `type` 字段区分三种子类型：

- `"checkpoint"` -- 来自 `checkpoints` 模式，step 使用 `self.step - 1`
- `"task"` -- 任务开始事件，step 使用 `self.step`
- `"task_result"` -- 任务完成事件，通过 `"result" in v` 判断

---

## 9.13 背压与缓冲

### 9.13.1 SyncQueue

同步流式使用 `SyncQueue`（基于 `queue.Queue`）：

```python
# libs/langgraph/langgraph/pregel/main.py

stream = SyncQueue()
```

`SyncQueue` 是一个无界队列，生产者（`_emit`、`StreamMessagesHandler`）直接 `put`，
消费者（`_output` 函数）通过 `get` 取出。

### 9.13.2 AsyncQueue

异步流式使用 `AsyncQueue`（基于 `asyncio.Queue`）：

```python
# libs/langgraph/langgraph/pregel/main.py  astream() 内

stream = AsyncQueue()
```

### 9.13.3 无界队列的背压考量

LangGraph 的流式系统没有显式的背压机制。队列是无界的，生产者永远不会被阻塞。
这是合理的设计选择，因为：

1. 流式事件通常很小（token、状态快照）
2. 消费者（`_output`）在每次 `runner.tick` yield 时都会排空队列
3. 真正的大量数据场景（如大型 messages 流）由 LLM 的速率自然限制

### 9.13.4 get_waiter 软同步

`get_waiter` 机制提供了一种软同步手段：

```python
def get_waiter() -> concurrent.futures.Future[None]:
    nonlocal waiter
    if waiter is None or waiter.done():
        waiter = loop.submit(stream.wait)
        return waiter
    else:
        return waiter
```

`stream.wait` 使用信号量阻塞直到有新数据到达。这让 `runner.tick` 在等待任务完成的
同时也能响应新的流式事件（如 LLM token），而不是纯粹等待 Future 完成。

### 9.13.5 退出时的信号量释放

```python
loop.stack.callback(stream._count.release)
```

当 loop 退出时，释放一次信号量。这确保正在等待的 `stream.wait` 能够返回，
避免 waiter Future 永久挂起。

---

## 9.14 v1 vs v2 输出格式

LangGraph 支持两种输出格式版本：

### v1（默认）

```python
# 单模式
for chunk in graph.stream(input):
    print(chunk)  # dict | Any

# 多模式
for mode, chunk in graph.stream(input, stream_mode=["values", "updates"]):
    print(mode, chunk)  # str, dict | Any

# 子图
for ns, chunk in graph.stream(input, subgraphs=True):
    print(ns, chunk)  # tuple, dict | Any
```

v1 的输出格式根据参数组合变化，比较灵活但类型不够精确。

### v2

```python
for part in graph.stream(input, version="v2"):
    # part 是 StreamPart (TypedDict)
    print(part["type"])  # "values" | "updates" | "messages" | ...
    print(part["ns"])    # tuple[str, ...]
    print(part["data"])  # 类型取决于 part["type"]
```

v2 通过 `StreamPart` 的 discriminated union 提供一致的输出结构，
类型更精确，适合 IDE 自动补全和类型检查。

v2 还支持 pydantic/dataclass 状态类型的自动映射：

```python
# libs/langgraph/langgraph/pregel/main.py

_output_mapper = self._output_mapper if version == "v2" else None
_state_mapper = self._state_mapper if version == "v2" else None
```

`_output_mapper` 和 `_state_mapper` 由 `CompiledStateGraph` 设置，将 dict 状态
转换为 Pydantic model 或 dataclass 实例。

---

## 9.15 Subgraphs 流式输出

当 `subgraphs=True` 时，子图的流式事件也会传递到父图：

```python
# libs/langgraph/langgraph/pregel/main.py  stream() 内

if subgraphs:
    loop.config[CONF][CONFIG_KEY_STREAM] = loop.stream
```

将 `loop.stream` 写入 config，子图的 `PregelLoop` 构造函数中会通过
`DuplexStream` 合并：

```python
# PregelLoop.__init__
if self.stream is not None and CONFIG_KEY_STREAM in config[CONF]:
    self.stream = DuplexStream(self.stream, config[CONF][CONFIG_KEY_STREAM])
```

子图发射的事件携带自己的 `checkpoint_ns`，消费者通过 `ns` 元组判断事件来源层级：

```python
# v1 + subgraphs 输出示例
for ns, mode, chunk in graph.stream(
    input, subgraphs=True, stream_mode=["updates"]
):
    if ns:
        print(f"Subgraph {'/'.join(ns)}: {chunk}")
    else:
        print(f"Main graph: {chunk}")
```

---

## 9.16 GraphOutput：invoke 的 v2 返回值

当使用 `version="v2"` 调用 `invoke` 时，返回 `GraphOutput` 对象：

```python
# libs/langgraph/langgraph/types.py

@dataclass(frozen=True)
class GraphOutput(Generic[OutputT]):
    value: OutputT
    interrupts: tuple[Interrupt, ...] = ()

    def __getitem__(self, key: str) -> Any:
        """Backward compat: result['key'] access."""
        warn("Accessing GraphOutput via result[key] is deprecated. ...")
        if key == _INTERRUPT_KEY:
            return self.interrupts
        if isinstance(self.value, dict):
            return self.value[key]
        try:
            return getattr(self.value, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        warn("Accessing GraphOutput via key in result is deprecated. ...")
        if key == _INTERRUPT_KEY:
            return bool(self.interrupts)
        if isinstance(self.value, dict):
            return key in self.value
        return isinstance(key, str) and hasattr(self.value, key)
```

`GraphOutput` 将输出值和中断信息分离，避免了 v1 中 `__interrupt__` 混在
输出字典中的问题。旧的 `result["key"]` 访问方式仍然支持但已标记为 deprecated。

---

## 9.17 print_mode：调试打印

```python
# libs/langgraph/langgraph/pregel/main.py  stream() 内

if debug or self.debug:
    print_mode = ["updates", "values"]
```

`print_mode` 指定哪些模式的输出应该打印到控制台。它不影响 `stream()` 的返回值，
只是一种调试辅助。`_output` 函数内部检查 print_mode 并调用打印逻辑。

---

## 9.18 错误处理与流式输出

当任务执行失败时：

```python
# output_writes() 中的判断
if writes[0][0] == INTERRUPT:
    ...  # interrupt 处理
elif writes[0][0] != ERROR:
    ...  # 正常更新
# ERROR 情况下：不发射 updates 事件

if not cached:
    self._emit("tasks", map_debug_task_results, ...)
    # task_result 事件中包含 error 字段
```

错误不会产生 `updates` 事件，但会产生 `tasks` 事件（包含 `error` 字段）。
这让监控系统可以跟踪失败，而 UI 不会收到无效的 updates。

---

## 9.19 流式事件的完整生命周期

以一次包含 LLM 调用的单步执行为例：

```
stream(input, stream_mode=["values", "updates", "messages"])
  |
  v
SyncPregelLoop.__enter__()
  |
  _first(): 处理输入
  |
  _emit("values", ...)  --> values 事件 #1（resume 时发射）
  |
  v
tick():
  prepare_next_tasks() -> {task_id: llm_node_task}
  |
  _emit("tasks", map_debug_tasks, ...)  --> tasks 事件（如果订阅了）
  |
  v
runner.tick():
  submit(llm_node_task)
  |
  --> llm_node 开始执行
  |
  --> ChatModel.stream() 开始
  |     |
  |     StreamMessagesHandler.on_chat_model_start()
  |     |   记录 run_id -> (ns, metadata)
  |     |
  |     StreamMessagesHandler.on_llm_new_token("Hello")
  |     |   stream.put((ns, "messages", (AIMessageChunk("Hello"), metadata)))
  |     |
  |     StreamMessagesHandler.on_llm_new_token(" world")
  |     |   stream.put((ns, "messages", (AIMessageChunk(" world"), metadata)))
  |     |
  |     StreamMessagesHandler.on_llm_end(response)
  |         stream.put((ns, "messages", (AIMessage("Hello world"), metadata)))
  |                                                [dedupe=True]
  |
  --> llm_node 返回 {"response": "Hello world"}
  |
  --> runner commit: put_writes(task_id, writes)
  |     |
  |     output_writes():
  |       _emit("updates", map_output_updates, ...)
  |       --> updates 事件: {"llm_node": {"response": "Hello world"}}
  |
  get_waiter 唤醒 -> yield
  |
  _output():  从队列取出所有事件
    --> yield messages 事件 (token "Hello")
    --> yield messages 事件 (token " world")
    --> yield messages 事件 (完整消息, dedupe 可能跳过)
    --> yield updates 事件
  |
  v
after_tick():
  apply_writes()
  _emit("values", map_output_values, ...)
    --> values 事件 #2: 完整状态快照
  |
  v
tick() -> False (no more tasks)
  |
  v
_output(): 取出剩余事件
  --> yield values 事件 #2
```

---

## 本章要点

1. **七种流式模式**：`values`（状态快照）、`updates`（增量更新）、`messages`（LLM token）、
   `custom`（自定义数据）、`checkpoints`（checkpoint 事件）、`tasks`（任务生命周期）、
   `debug`（全面调试）。每种模式对应精确的 TypedDict 类型。

2. **StreamPart 类型系统**：v2 API 通过 discriminated union `StreamPart` 提供类型安全的
   流式输出。`part["type"]` 字段用于类型窄化，支持 IDE 自动补全。

3. **StreamMessagesHandler**：messages 模式通过 LangChain callback 系统实现。
   `on_llm_new_token` 拦截 LLM token 流，`on_chain_end` 从节点输出提取消息。
   `seen` 集合实现去重，避免重复发送已有消息。

4. **DuplexStream**：子图的流式事件通过 DuplexStream 向上传播到父图。
   每个事件携带 namespace 标识来源层级。

5. **背压与缓冲**：使用无界队列（SyncQueue/AsyncQueue），生产者不阻塞。
   `get_waiter` 机制通过信号量实现软同步，让主循环在等待任务完成时也能响应新数据。

6. **事件发射时机**：values 在超步结束/interrupt/resume 时发射；updates 在每个任务完成时发射；
   messages 在每个 LLM token 和节点输出时发射；tasks 在任务开始和完成时发射。

7. **debug 复用**：debug 模式不生成独立事件，而是在 checkpoints 和 tasks 事件上
   添加 step/timestamp/type 包装，避免重复计算。
