# 第 14 章 Interrupt 与 Human-in-the-Loop

在 AI Agent 的实际部署中，"让人类参与决策"是一项刚需。LangGraph 通过 `interrupt()` 函数与 `Command(resume=...)` 机制，提供了一套完整的 Human-in-the-Loop 方案。本章将从源码层面深入分析 interrupt 的实现原理、执行循环中的处理逻辑、恢复机制以及实战模式。

## interrupt() 函数：在节点执行中暂停

`interrupt()` 是 LangGraph 暴露给用户的核心 API，定义在 `langgraph/types.py` 中。它的作用是在节点执行过程中暂停图的运行，并将一个值传递给客户端。

```python
# libs/langgraph/langgraph/types.py
def interrupt(value: Any) -> Any:
    from langgraph._internal._constants import (
        CONFIG_KEY_CHECKPOINT_NS,
        CONFIG_KEY_SCRATCHPAD,
        CONFIG_KEY_SEND,
        RESUME,
    )
    from langgraph.config import get_config
    from langgraph.errors import GraphInterrupt

    conf = get_config()["configurable"]
    # track interrupt index
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()
    # find previous resume values
    if scratchpad.resume:
        if idx < len(scratchpad.resume):
            conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
            return scratchpad.resume[idx]
    # find current resume value
    v = scratchpad.get_null_resume(True)
    if v is not None:
        assert len(scratchpad.resume) == idx, (scratchpad.resume, idx)
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return v
    # no resume value found
    raise GraphInterrupt(
        (
            Interrupt.from_ns(
                value=value,
                ns=conf[CONFIG_KEY_CHECKPOINT_NS],
            ),
        )
    )
```

这段代码的执行逻辑分为三个阶段：

### 阶段一：首次调用 -- 抛出 GraphInterrupt

当节点第一次执行到 `interrupt()` 时，scratchpad 中没有任何 resume 值。函数会走到最后的 `raise GraphInterrupt(...)` 分支。`GraphInterrupt` 是一个特殊异常，它携带一个 `Interrupt` 对象的元组。

`Interrupt` 数据类本身非常简洁：

```python
# libs/langgraph/langgraph/types.py
@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any
    """The value associated with the interrupt."""
    id: str
    """The ID of the interrupt. Can be used to resume the interrupt directly."""

    @classmethod
    def from_ns(cls, value: Any, ns: str) -> Interrupt:
        return cls(value=value, id=xxh3_128_hexdigest(ns.encode()))
```

`Interrupt.id` 通过对 `checkpoint_ns` 做 xxh3 hash 生成，保证了同一 namespace 下的 interrupt 具有稳定的 ID。这个 ID 在恢复时用于精确匹配。

### 阶段二：恢复时 -- 返回 resume 值

当用户通过 `Command(resume=value)` 恢复执行时，节点会**从头重新执行**。再次碰到 `interrupt()` 调用时，scratchpad 中已经有了 resume 值列表。函数检查当前的 interrupt 索引（`idx`）是否在 resume 列表范围内，如果在则直接返回对应的值。

### 阶段三：多个 interrupt 的顺序匹配

如果一个节点包含多个 `interrupt()` 调用，LangGraph 通过 `scratchpad.interrupt_counter()` 维护一个递增计数器，按顺序匹配 resume 值与 interrupt 调用。这意味着 resume 值列表与 interrupt 调用的顺序必须一一对应。

```python
# 节点中包含多个 interrupt 的场景
def review_node(state):
    name = interrupt("请输入你的名字")       # idx=0
    feedback = interrupt("请输入你的反馈")    # idx=1
    return {"name": name, "feedback": feedback}
```

## INTERRUPT 常量在执行循环中的处理

在内部常量系统中，`INTERRUPT` 是一个 reserved write key：

```python
# libs/langgraph/langgraph/_internal/_constants.py
INTERRUPT = sys.intern("__interrupt__")
# for dynamic interrupts raised by nodes
RESUME = sys.intern("__resume__")
# for values passed to resume a node after an interrupt
```

执行循环 `PregelLoop`（定义在 `langgraph/pregel/_loop.py`）在两个关键时机检查是否需要中断。

### interrupt_before：执行前检查

```python
# libs/langgraph/langgraph/pregel/_loop.py
# before execution, check if we should interrupt
if self.interrupt_before and should_interrupt(
    self.checkpoint, self.interrupt_before, self.tasks.values()
):
    self.status = "interrupt_before"
    raise GraphInterrupt()
```

### interrupt_after：执行后检查

```python
# libs/langgraph/langgraph/pregel/_loop.py
def after_tick(self) -> None:
    # ... apply writes, produce values output ...
    # save checkpoint
    self._put_checkpoint({"source": "loop"})
    # after execution, check if we should interrupt
    if self.interrupt_after and should_interrupt(
        self.checkpoint, self.interrupt_after, self.tasks.values()
    ):
        self.status = "interrupt_after"
        raise GraphInterrupt()
```

`should_interrupt()` 函数定义在 `_algo.py` 中，它的判断逻辑基于 channel version：

```python
# libs/langgraph/langgraph/pregel/_algo.py
def should_interrupt(
    checkpoint: Checkpoint,
    interrupt_nodes: All | Sequence[str],
    tasks: Iterable[PregelExecutableTask],
) -> list[PregelExecutableTask]:
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()
    seen = checkpoint["versions_seen"].get(INTERRUPT, {})
    # interrupt if any channel has been updated since last interrupt
    any_updates_since_prev_interrupt = any(
        version > seen.get(chan, null_version)
        for chan, version in checkpoint["channel_versions"].items()
    )
    # and any triggered node is in interrupt_nodes list
    return (
        [
            task
            for task in tasks
            if (interrupt_nodes == "*" or task.name in interrupt_nodes)
        ]
        if any_updates_since_prev_interrupt
        else []
    )
```

这里的关键设计是通过 `versions_seen[INTERRUPT]` 跟踪上次中断时各 channel 的版本号，确保同一状态不会重复触发中断。

### pending writes 中的 INTERRUPT 处理

在 `_match_writes` 方法中，INTERRUPT 和 RESUME 类型的 pending writes 会被跳过，不会被应用到 task 的 writes 中：

```python
# libs/langgraph/langgraph/pregel/_loop.py
def _match_writes(self, tasks: Mapping[str, PregelExecutableTask]) -> None:
    for tid, k, v in self.checkpoint_pending_writes:
        if k in (ERROR, INTERRUPT, RESUME):
            continue
        if task := tasks.get(tid):
            task.writes.append((k, v))
```

`_pending_interrupts()` 方法追踪哪些 interrupt 还没有被 resume：

```python
# libs/langgraph/langgraph/pregel/_loop.py
def _pending_interrupts(self) -> set[str]:
    pending_interrupts: dict[str, str] = {}
    pending_resumes: set[str] = set()

    for task_id, write_type, value in self.checkpoint_pending_writes:
        if write_type == INTERRUPT:
            pending_interrupts[task_id] = value[0].id
        elif write_type == RESUME:
            pending_resumes.add(task_id)

    resumed_interrupt_ids = {
        pending_interrupts[task_id]
        for task_id in pending_resumes
        if task_id in pending_interrupts
    }

    hanging_interrupts: set[str] = {
        interrupt_id
        for interrupt_id in pending_interrupts.values()
        if interrupt_id not in resumed_interrupt_ids
    }
    return hanging_interrupts
```

## 恢复：Command(resume=value) 的 Resume 机制

恢复中断的入口是 `Command` 类。当用户传入 `Command(resume=value)` 作为图的输入时，执行循环在 `_first()` 方法中处理这个 Command：

```python
# libs/langgraph/langgraph/pregel/_loop.py
if input_is_command:
    if (resume := cast(Command, self.input).resume) is not None:
        if not self.checkpointer:
            raise RuntimeError(
                "Cannot use Command(resume=...) without checkpointer"
            )

        if resume_is_map := (
            isinstance(resume, dict)
            and all(is_xxh3_128_hexdigest(k) for k in resume)
        ):
            self.config[CONF][CONFIG_KEY_RESUME_MAP] = resume
        else:
            if len(self._pending_interrupts()) > 1:
                raise RuntimeError(
                    "When there are multiple pending interrupts, you must specify "
                    "the interrupt id when resuming."
                )
```

这里有两种 resume 模式：

1. **单值 resume**：`Command(resume="some value")`。当只有一个 pending interrupt 时使用。
2. **Map resume**：`Command(resume={"interrupt_id_1": value1, "interrupt_id_2": value2})`。当有多个 pending interrupt 时，通过 interrupt ID 精确指定每个 interrupt 的 resume 值。

Command 输入随后被 `map_command()` 函数转换为 pending writes：

```python
# libs/langgraph/langgraph/pregel/_io.py
def map_command(cmd: Command) -> Iterator[tuple[str, str, Any]]:
    if cmd.goto:
        # ... handle goto ...
    if cmd.resume is not None:
        yield (NULL_TASK_ID, RESUME, cmd.resume)
    if cmd.update:
        for k, v in cmd._update_as_tuples():
            yield (NULL_TASK_ID, k, v)
```

resume 值被写入 checkpoint 的 pending writes 后，当执行循环恢复图的运行时，节点会重新执行。`interrupt()` 函数会从 scratchpad 中找到对应的 resume 值并返回，而不是再次抛出异常。

### 恢复时的 versions_seen 更新

在 `_first()` 方法中，恢复执行时会更新 `versions_seen[INTERRUPT]`，将所有 channel 的当前版本标记为已见：

```python
# libs/langgraph/langgraph/pregel/_loop.py
if is_resuming:
    self.checkpoint["versions_seen"].setdefault(INTERRUPT, {})
    for k in self.channels:
        if k in self.checkpoint["channel_versions"]:
            version = self.checkpoint["channel_versions"][k]
            self.checkpoint["versions_seen"][INTERRUPT][k] = version
```

这确保了恢复后不会立即再次触发 `interrupt_before` 或 `interrupt_after` 检查。

## prebuilt interrupt.py 的 HumanInterruptConfig

LangGraph 在 `langgraph/prebuilt/interrupt.py` 中提供了一套结构化的 Human-in-the-Loop 类型定义（目前已标记为 deprecated，迁移至 `langchain.agents.interrupt`）：

```python
# libs/prebuilt/langgraph/prebuilt/interrupt.py
class HumanInterruptConfig(TypedDict):
    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool

class ActionRequest(TypedDict):
    action: str
    args: dict

class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str | None

class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: None | str | ActionRequest
```

这套类型定义了人类交互的四种操作：
- **accept**：批准当前状态
- **ignore**：跳过当前步骤
- **respond**：提供文字反馈
- **edit**：编辑当前内容

使用示例：

```python
from langgraph.types import interrupt, Command

def tool_approval_node(state):
    request = HumanInterrupt(
        action_request=ActionRequest(
            action="run_command",
            args={"command": "rm -rf /tmp/cache"}
        ),
        config=HumanInterruptConfig(
            allow_ignore=True,
            allow_respond=True,
            allow_edit=False,
            allow_accept=True,
        ),
        description="请审批以下命令的执行"
    )
    response = interrupt([request])[0]
    if response["type"] == "accept":
        execute_command(state)
    elif response["type"] == "ignore":
        pass
```

## 实战模式

### 模式一：审批模式

最常见的 Human-in-the-Loop 场景。在执行敏感操作前暂停，等待人类批准。

```python
from langgraph.graph import StateGraph, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    action: str
    approved: bool

def propose_action(state):
    return {"action": "delete_database"}

def approval_gate(state):
    answer = interrupt(f"是否批准执行: {state['action']}?")
    return {"approved": answer == "yes"}

def execute_action(state):
    if state["approved"]:
        # 执行操作
        pass
    return state

builder = StateGraph(State)
builder.add_node("propose", propose_action)
builder.add_node("approve", approval_gate)
builder.add_node("execute", execute_action)
builder.add_edge(START, "propose")
builder.add_edge("propose", "approve")
builder.add_edge("approve", "execute")

graph = builder.compile(checkpointer=InMemorySaver())
```

### 模式二：编辑模式

让人类在继续之前修改中间结果。

```python
def draft_node(state):
    draft = generate_draft(state["topic"])
    edited = interrupt({"draft": draft, "instruction": "请编辑以下草稿"})
    return {"content": edited}
```

### 模式三：多轮对话

在一个节点中使用多个 `interrupt()` 实现交互式会话。

```python
def interactive_node(state):
    name = interrupt("请输入你的名字")
    preference = interrupt(f"你好 {name}，请选择你的偏好：A 或 B")
    return {"name": name, "preference": preference}

# 恢复时需要依次提供值
# 第一次 resume：Command(resume="Alice")
# 第二次 resume：Command(resume="A")
```

需要注意，每次 resume 节点都会从头重新执行。第一次 resume 后，`interrupt("请输入你的名字")` 返回 "Alice"，然后执行到第二个 `interrupt()` 时再次暂停。

### 模式四：compile 参数 interrupt_before / interrupt_after

除了在节点内调用 `interrupt()` 函数，还可以在编译图时通过参数指定在哪些节点前后自动中断：

```python
graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["sensitive_node"],
    interrupt_after=["review_node"],
)
```

`interrupt_before=["sensitive_node"]` 表示在 `sensitive_node` 执行前暂停。用户可以通过 `get_state()` 查看当前状态，再通过 `invoke(None, config)` 继续执行。设置 `interrupt_before="*"` 或 `interrupt_after="*"` 可以在所有节点前后暂停。

## _suppress_interrupt：顶层图与子图的差异化处理

当 `GraphInterrupt` 异常沿调用栈向上传播时，`_suppress_interrupt` 方法决定是否抑制该异常。这是 interrupt 机制中最关键的边界处理逻辑。

```python
# libs/langgraph/langgraph/pregel/_loop.py
def _suppress_interrupt(
    self,
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    traceback: TracebackType | None,
) -> bool | None:
    # persist current checkpoint and writes
    if self.durability == "exit" and (
        not self.is_nested
        or exc_value is not None
        or all(NS_END not in part for part in self.checkpoint_ns)
    ):
        self._put_checkpoint(self.checkpoint_metadata)
        self._put_pending_writes()
    # suppress interrupt
    suppress = isinstance(exc_value, GraphInterrupt) and not self.is_nested
    if suppress:
        # emit one last "values" event, with pending writes applied
        if (
            hasattr(self, "tasks")
            and self.checkpoint_pending_writes
            and any(task.writes for task in self.tasks.values())
        ):
            updated_channels = apply_writes(
                self.checkpoint,
                self.channels,
                self.tasks.values(),
                self.checkpointer_get_next_version,
                self.trigger_to_nodes,
            )
            if not updated_channels.isdisjoint(
                (self.output_keys,)
                if isinstance(self.output_keys, str)
                else self.output_keys
            ):
                self._emit(
                    "values",
                    map_output_values,
                    self.output_keys,
                    [w for t in self.tasks.values() for w in t.writes],
                    self.channels,
                )
        # emit INTERRUPT if exception is empty
        if exc_value is not None and (not exc_value.args or not exc_value.args[0]):
            self._emit(
                "updates",
                lambda: iter(
                    [{INTERRUPT: cast(GraphInterrupt, exc_value).args[0]}]
                ),
            )
        # save final output
        self.output = read_channels(self.channels, self.output_keys)
        return True
    elif exc_type is None:
        self.output = read_channels(self.channels, self.output_keys)
```

关键行为总结：

1. **`suppress = isinstance(exc_value, GraphInterrupt) and not self.is_nested`**：只有**根图**才抑制 `GraphInterrupt`。子图（`is_nested=True`）不抑制，让异常继续向上传播。

2. **在抑制前发射最后的 stream 事件**：包括 "values" 事件（带 pending writes 应用后的最新状态）和 "updates" 事件（带 `__interrupt__` 键）。

3. **保存最终输出**：将当前 channel 值读取为输出，确保调用者可以获取到中断时的状态。

4. **`durability="exit"` 模式下的持久化**：在退出时持久化 checkpoint 和 pending writes，而非每步都持久化。

## output_writes 中的 INTERRUPT 输出

当节点写入了 INTERRUPT 类型的数据时，`output_writes` 方法负责将其通过 stream 发送给客户端：

```python
# libs/langgraph/langgraph/pregel/_loop.py
def output_writes(
    self, task_id: str, writes: WritesT, *, cached: bool = False
) -> None:
    if task := self.tasks.get(task_id):
        if task.config is not None and TAG_HIDDEN in task.config.get(
            "tags", EMPTY_SEQ
        ):
            return
        if writes[0][0] == INTERRUPT:
            # PUSH task 中如果有 call 标记，不直接 emit（由父级处理）
            if task.path[0] == PUSH and task.path[-1] is True:
                return
            interrupts = [
                {
                    INTERRUPT: tuple(
                        v
                        for w in writes
                        if w[0] == INTERRUPT
                        for v in (w[1] if isinstance(w[1], Sequence) else (w[1],))
                    )
                }
            ]
            stream_modes = self.stream.modes if self.stream else []
            if "updates" in stream_modes:
                self._emit("updates", lambda: iter(interrupts))
            if "values" in stream_modes:
                current_values = read_channels(self.channels, self.output_keys)
                if isinstance(current_values, dict):
                    current_values[INTERRUPT] = interrupts[0][INTERRUPT]
                    self._emit("values", lambda: iter([current_values]))
                else:
                    self._emit("values", lambda: iter(interrupts))
```

注意对 PUSH task 的特殊处理：如果 task 是由 `call()` 创建的（`task.path[-1] is True`），则不直接发射 interrupt 事件——因为 interrupt 会由包含该 call 的外层 task 处理。

## GraphInterrupt 的传播路径

让我们追踪一个完整的 interrupt 从抛出到客户端接收的完整路径：

```
节点代码调用 interrupt("请审批")
    |
    v
interrupt() 函数检查 scratchpad，没有 resume 值
    |
    v
抛出 GraphInterrupt((Interrupt(value="请审批", id="abc123"),))
    |
    v
Pregel runner 捕获异常
    |
    v
将 (task_id, INTERRUPT, (Interrupt(...),)) 写入 pending_writes
    |
    v
调用 output_writes() 发射 stream 事件
    |
    v
PregelLoop._suppress_interrupt() 被调用
    |--- 如果是子图 (is_nested=True)：异常继续向上传播
    |--- 如果是根图 (is_nested=False)：
    |    |--- 应用 pending writes 到 channels
    |    |--- 发射 "values" 事件（带最新状态）
    |    |--- 发射 "updates" 事件（带 __interrupt__ 键）
    |    |--- 保存最终输出
    |    `--- 返回 True（抑制异常）
    v
客户端从 stream 接收到 interrupt 事件
```

## StateSnapshot 中的 interrupts

当客户端调用 `get_state()` 时，可以查看当前的 interrupt 状态：

```python
# libs/langgraph/langgraph/types.py
class StateSnapshot(NamedTuple):
    values: dict[str, Any] | Any
    next: tuple[str, ...]
    config: RunnableConfig
    metadata: CheckpointMetadata | None
    created_at: str | None
    parent_config: RunnableConfig | None
    tasks: tuple[PregelTask, ...]
    interrupts: tuple[Interrupt, ...]
```

`interrupts` 字段聚合了所有 task 中的 interrupt。`PregelTask` 本身也包含 interrupts：

```python
class PregelTask(NamedTuple):
    id: str
    name: str
    path: tuple[str | int | tuple, ...]
    error: Exception | None = None
    interrupts: tuple[Interrupt, ...] = ()
    state: None | RunnableConfig | StateSnapshot = None
    result: Any | None = None
```

客户端可以通过以下方式检查和处理 interrupt：

```python
state = graph.get_state(config)

# 检查是否有 pending interrupt
if state.interrupts:
    for intr in state.interrupts:
        print(f"Interrupt: {intr.value} (id={intr.id})")

# 也可以通过 task 级别查看
for task in state.tasks:
    if task.interrupts:
        print(f"Task {task.name} interrupted: {task.interrupts}")
```

## RESUME 写入的清理与保留策略

在时间旅行（从特定 checkpoint 重放）场景下，RESUME 写入需要特殊处理：

```python
# libs/langgraph/langgraph/pregel/_loop.py
# When replaying from a specific checkpoint, drop cached RESUME
# writes so that interrupt() calls re-fire instead of returning
# stale values. But if we're actively resuming, keep them --
# multi-interrupt scenarios need previously resolved values preserved.
if self.is_replaying and not (
    (input_is_command and cast(Command, self.input).resume is not None)
    or configurable.get(CONFIG_KEY_RESUMING, False)
):
    self.checkpoint_pending_writes = [
        w for w in self.checkpoint_pending_writes if w[1] != RESUME
    ]
```

这段逻辑的两个分支：

- **重放但非 resume**（`is_replaying=True`, 无 resume 值）：清除所有 RESUME 写入。这样 `interrupt()` 调用会重新触发，让用户重新做决策。
- **正在 resume**（有 resume 值或 `CONFIG_KEY_RESUMING=True`）：保留 RESUME 写入。在多 interrupt 场景中，之前已经解决的 interrupt 的 resume 值需要保留，否则会丢失。

## 声明式中断 vs 动态中断的对比

| 特性 | `interrupt()` 函数 | `interrupt_before`/`interrupt_after` |
|------|-------------------|--------------------------------------|
| 定义位置 | 节点函数内部 | `compile()` 参数 |
| 值传递 | 可传递任意值给客户端 | 不传递值，仅暂停执行 |
| 恢复方式 | `Command(resume=value)` | `invoke(None, config)` |
| 条件性 | 运行时动态决定 | 编译时静态声明 |
| 粒度 | 节点内任意位置 | 节点执行前/后 |
| 多次触发 | 一个节点可多次 interrupt | 每个 superstep 最多一次 |
| ID 机制 | 每个 interrupt 有唯一 ID | 无 ID |

## Checkpointer 的关键角色

interrupt 机制的每一个环节都依赖 checkpointer：

```python
if (resume := cast(Command, self.input).resume) is not None:
    if not self.checkpointer:
        raise RuntimeError(
            "Cannot use Command(resume=...) without checkpointer"
        )
```

Checkpointer 在 interrupt 流程中的具体作用：

1. **中断时保存状态**：channel 值、pending writes（包含 INTERRUPT 写入）全部持久化
2. **保存 resume 值**：`Command(resume=...)` 产生的 RESUME 写入被持久化
3. **恢复执行**：从 checkpoint 加载完整状态，重建 channel 和 task
4. **版本追踪**：`versions_seen[INTERRUPT]` 记录了上次中断时的 channel 版本，防止重复中断
5. **多 interrupt 协调**：通过 pending writes 中的 INTERRUPT 和 RESUME 记录追踪各个 interrupt 的状态

## 本章要点

1. **`interrupt(value)` 的双重角色**：首次调用时抛出 `GraphInterrupt` 异常暂停执行，恢复后重新执行时返回 resume 值。这种"抛出-重放"的设计使得 interrupt 可以嵌入到任何节点逻辑中。

2. **基于计数器的顺序匹配**：同一节点中的多个 `interrupt()` 通过 `scratchpad.interrupt_counter()` 按顺序与 resume 值一一对应，无需额外标识。

3. **两种中断机制**：`interrupt()` 函数是动态中断，由节点逻辑控制；`interrupt_before` / `interrupt_after` 是声明式中断，由编译参数控制，通过 `should_interrupt()` 和 channel version 比较实现。

4. **resume 的两种模式**：单值模式适用于单个 pending interrupt；Map 模式（以 interrupt ID 为 key）适用于多个并发 interrupt 的场景。

5. **Checkpoint 依赖**：interrupt 机制强依赖 checkpointer。没有 checkpointer 时使用 `Command(resume=...)` 会直接抛出 `RuntimeError`。

6. **HumanInterruptConfig 结构化交互**：prebuilt 模块提供了标准化的人机交互类型定义，将交互抽象为 accept / ignore / respond / edit 四种操作。

7. **根图抑制 GraphInterrupt，子图传播**：`_suppress_interrupt` 中通过 `is_nested` 标志区分处理。根图在抑制前发射最后的 stream 事件并保存输出。

8. **`_pending_interrupts()` 追踪未解决的中断**：通过比对 INTERRUPT 写入和 RESUME 写入，确定哪些 interrupt 还需要 resume。

9. **时间旅行中的 RESUME 清理**：重放时清除旧的 RESUME 写入让 interrupt 重新触发，但 resume 时保留已解决的值。

10. **output_writes 对 PUSH task 的特殊处理**：由 `call()` 创建的 task 不直接发射 interrupt 事件，避免重复发射。
