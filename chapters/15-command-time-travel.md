# 第 15 章 Command 与时间旅行

`Command` 是 LangGraph 中一个多功能的控制原语。它不仅用于恢复 interrupt（如上一章所述），还提供了状态更新、路由跳转、消息发送等能力。结合 checkpointer 的状态历史功能，`Command` 还是实现"时间旅行"的核心工具。本章将全面分析 `Command` 的四个字段、在执行循环中的路由逻辑，以及 `get_state_history` / Fork 等时间旅行机制。

## Command 类：四个字段

`Command` 定义在 `langgraph/types.py` 中，是一个泛型 frozen dataclass：

```python
# libs/langgraph/langgraph/types.py
@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    graph: str | None = None
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()

    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"
```

四个字段各有分工：

### update：状态更新

`update` 字段用于更新图的状态。它支持 dict、tuple 列表、Pydantic model 等多种格式。内部通过 `_update_as_tuples()` 统一转换为 `(key, value)` 元组列表：

```python
# libs/langgraph/langgraph/types.py
def _update_as_tuples(self) -> Sequence[tuple[str, Any]]:
    if isinstance(self.update, dict):
        return list(self.update.items())
    elif isinstance(self.update, (list, tuple)) and all(
        isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], str)
        for t in self.update
    ):
        return self.update
    elif keys := get_cached_annotated_keys(type(self.update)):
        return get_update_as_tuples(self.update, keys)
    elif self.update is not None:
        return [("__root__", self.update)]
    else:
        return []
```

这段代码兼容了四种输入格式：普通 dict、元组列表、带注解字段的对象（Pydantic model / dataclass），以及单个值（映射到 `__root__`）。

### goto：路由跳转

`goto` 字段指定下一步要执行的节点。它可以是节点名（string）、`Send` 对象、或它们的列表。`Send` 允许携带自定义输入发送给目标节点：

```python
# libs/langgraph/langgraph/types.py
class Send:
    __slots__ = ("node", "arg")

    node: str
    arg: Any

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg
```

### resume：恢复中断

如上一章所述，`resume` 可以是单个值（当只有一个 pending interrupt 时）或一个以 interrupt ID 为 key 的 dict（多个 pending interrupt 时）。

### graph：目标图

`graph` 字段指定 Command 应用的目标图。`None` 表示当前图，`Command.PARENT`（即 `"__parent__"`）表示最近的父图。这在子图中向父图发送更新时非常有用。

## 节点返回 Command 时的路由逻辑

当节点返回一个 `Command` 对象时，`map_command()` 函数将其转换为执行循环可以处理的 writes：

```python
# libs/langgraph/langgraph/pregel/_io.py
def map_command(cmd: Command) -> Iterator[tuple[str, str, Any]]:
    """Map input chunk to a sequence of pending writes in the form (channel, value)."""
    if cmd.graph == Command.PARENT:
        raise InvalidUpdateError("There is no parent graph")
    if cmd.goto:
        if isinstance(cmd.goto, (tuple, list)):
            sends = cmd.goto
        else:
            sends = [cmd.goto]
        for send in sends:
            if isinstance(send, Send):
                yield (NULL_TASK_ID, TASKS, send)
            elif isinstance(send, str):
                yield (NULL_TASK_ID, f"branch:to:{send}", START)
            else:
                raise TypeError(
                    f"In Command.goto, expected Send/str, got {type(send).__name__}"
                )
    if cmd.resume is not None:
        yield (NULL_TASK_ID, RESUME, cmd.resume)
    if cmd.update:
        for k, v in cmd._update_as_tuples():
            yield (NULL_TASK_ID, k, v)
```

这里有几个重要的映射规则：

1. **字符串 goto**：生成一个 `f"branch:to:{send}"` channel write，值为 `START`。这与条件边的 branch 机制对接，触发目标节点的执行。
2. **Send 对象**：写入 `TASKS` channel，等效于在条件边中返回 `Send` 对象。
3. **update**：将更新拆分为独立的 channel writes。
4. **resume**：写入 `RESUME` channel。

当 Command 作为图的输入时（而非节点返回值），处理逻辑在 `_first()` 方法中：

```python
# libs/langgraph/langgraph/pregel/_loop.py
input_is_command = isinstance(self.input, Command)
is_resuming = bool(self.checkpoint["channel_versions"]) and bool(
    configurable.get(
        CONFIG_KEY_RESUMING,
        self.input is None
        or input_is_command
        or (
            not self.is_nested
            and self.config.get("metadata", {}).get("run_id")
            == self.checkpoint_metadata.get("run_id", MISSING)
        ),
    )
)
```

执行循环将 Command 输入视为"恢复"操作的一种形式。`is_resuming` 为 `True` 意味着执行循环会从上次 checkpoint 继续，而不是从头开始。

## 时间旅行：get_state_history

LangGraph 的 checkpointer 保存了图执行过程中每一步的完整状态快照。通过 `get_state_history()` 可以回溯这些历史状态：

```python
# libs/langgraph/langgraph/pregel/main.py
def get_state_history(
    self,
    config: RunnableConfig,
    *,
    filter: dict[str, Any] | None = None,
    before: RunnableConfig | None = None,
    limit: int | None = None,
) -> Iterator[StateSnapshot]:
    """Get the history of the state of the graph."""
    config = ensure_config(config)
    checkpointer = config[CONF].get(
        CONFIG_KEY_CHECKPOINTER, self.checkpointer
    )
    if not checkpointer:
        raise ValueError("No checkpointer set")

    if (
        checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
    ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
        recast = recast_checkpoint_ns(checkpoint_ns)
        for _, pregel in self.get_subgraphs(namespace=recast, recurse=True):
            yield from pregel.get_state_history(
                patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                filter=filter,
                before=before,
                limit=limit,
            )
```

`get_state_history()` 返回一个 `StateSnapshot` 迭代器，按时间倒序排列。每个 `StateSnapshot` 包含了该时刻的完整信息：

```python
# libs/langgraph/langgraph/types.py
class StateSnapshot(NamedTuple):
    values: dict[str, Any] | Any
    """Current values of channels."""
    next: tuple[str, ...]
    """The name of the node to execute in each task for this step."""
    config: RunnableConfig
    """Config used to fetch this snapshot."""
    metadata: CheckpointMetadata | None
    """Metadata associated with this snapshot."""
    created_at: str | None
    """Timestamp of snapshot creation."""
    parent_config: RunnableConfig | None
    """Config used to fetch the parent snapshot, if any."""
    tasks: tuple[PregelTask, ...]
    """Tasks to execute in this step. If already attempted, may contain an error."""
    interrupts: tuple[Interrupt, ...]
    """Interrupts that occurred in this step that are pending resolution."""
```

### get_state：获取当前状态

`get_state()` 返回图的当前状态快照。它的实现调用了 checkpointer 的 `get_tuple()` 方法获取最新的 checkpoint，然后通过 `_prepare_state_snapshot()` 构建 `StateSnapshot`：

```python
# libs/langgraph/langgraph/pregel/main.py
def get_state(
    self, config: RunnableConfig, *, subgraphs: bool = False
) -> StateSnapshot:
    """Get the current state of the graph."""
    checkpointer = ensure_config(config)[CONF].get(
        CONFIG_KEY_CHECKPOINTER, self.checkpointer
    )
    if not checkpointer:
        raise ValueError("No checkpointer set")
    # ...
    saved = checkpointer.get_tuple(config)
    return self._prepare_state_snapshot(
        config, saved,
        recurse=checkpointer if subgraphs else None,
    )
```

`_prepare_state_snapshot()` 的核心逻辑是重建 channels、准备 next_tasks，并收集子图状态：

```python
# libs/langgraph/langgraph/pregel/main.py
def _prepare_state_snapshot(
    self,
    config: RunnableConfig,
    saved: CheckpointTuple | None,
    recurse: BaseCheckpointSaver | None = None,
) -> StateSnapshot:
    if not saved:
        return StateSnapshot(
            values={}, next=(), config=config,
            metadata=None, created_at=None, parent_config=None,
            tasks=(), interrupts=(),
        )

    step = saved.metadata.get("step", -1) + 1
    channels, managed = channels_from_checkpoint(
        self.channels, saved.checkpoint,
    )
    next_tasks = prepare_next_tasks(
        saved.checkpoint, saved.pending_writes or [],
        self.nodes, channels, managed, saved.config,
        step, step + 2,
        for_execution=True, store=self.store,
        checkpointer=self.checkpointer, manager=None,
    )
    # get the subgraphs
    subgraphs = dict(self.get_subgraphs())
    parent_ns = saved.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
    task_states: dict[str, RunnableConfig | StateSnapshot] = {}
    for task in next_tasks.values():
        if task.name not in subgraphs:
            continue
        task_ns = f"{task.name}{NS_END}{task.id}"
        if parent_ns:
            task_ns = f"{parent_ns}{NS_SEP}{task_ns}"
        # ...
```

## Fork：从历史状态分支执行

时间旅行的核心用途之一是 Fork --从历史状态的某个点分支出新的执行路径。实现方式是从 `get_state_history()` 获取目标状态的 `config`，然后用该 config 恢复执行：

```python
# 获取历史状态
history = list(graph.get_state_history(config))

# 选择一个历史状态（比如第 3 步）
target_state = history[2]

# 从该状态分支执行
forked_config = target_state.config
result = graph.invoke(
    Command(update={"key": "new_value"}),
    config=forked_config
)
```

也可以通过直接指定 `checkpoint_id` 来回到特定状态：

```python
config_at_checkpoint = {
    "configurable": {
        "thread_id": "my-thread",
        "checkpoint_id": "1ef5e3c5-a9c0-6d30-8003-1a15b3a9a4b0",
    }
}
result = graph.invoke(None, config=config_at_checkpoint)
```

当 `is_replaying` 为 `True` 时（即指定了 `checkpoint_id`），执行循环会进入重放模式。在这个模式下：

```python
# libs/langgraph/langgraph/pregel/_loop.py
if self.is_replaying and not (
    (input_is_command and cast(Command, self.input).resume is not None)
    or configurable.get(CONFIG_KEY_RESUMING, False)
):
    self.checkpoint_pending_writes = [
        w for w in self.checkpoint_pending_writes if w[1] != RESUME
    ]
```

重放模式会清除之前的 RESUME writes，让 `interrupt()` 调用重新触发，而不是返回旧的 resume 值。但如果同时提供了新的 resume 值，则保留这些 writes。

## update_state：外部修改状态

`update_state()` 允许从外部修改图的当前状态，模拟某个节点的写入：

```python
# libs/langgraph/langgraph/pregel/main.py
def update_state(
    self,
    config: RunnableConfig,
    values: dict[str, Any] | Any | None,
    as_node: str | None = None,
    task_id: str | None = None,
) -> RunnableConfig:
    """Update the state of the graph with the given values, as if they came from
    node `as_node`. If `as_node` is not provided, it will be set to the last node
    that updated the state, if not ambiguous.
    """
    return self.bulk_update_state(config, [[StateUpdate(values, as_node, task_id)]])
```

`update_state()` 与 `Command(update=...)` 的区别在于：前者在图执行外部调用，直接修改 checkpoint；后者在图执行内部（作为输入或节点返回值）使用，通过执行循环的正常流程处理。

## Command 的组合使用

`Command` 的四个字段可以组合使用，实现复杂的控制流。例如，在恢复 interrupt 的同时更新状态并跳转到指定节点：

```python
# 恢复中断，同时更新状态，并跳转到指定节点
command = Command(
    resume="approved",
    update={"audit_log": "approved by admin"},
    goto="execute_node"
)
graph.invoke(command, config)
```

又如，在节点内部返回 Command 实现动态路由：

```python
def router_node(state):
    if state["score"] > 0.8:
        return Command(
            update={"status": "high_quality"},
            goto="publish"
        )
    else:
        return Command(
            update={"status": "needs_review"},
            goto=Send("review", {"content": state["content"]})
        )
```

### 向父图发送 Command

在子图中，可以通过 `Command(graph=Command.PARENT)` 向父图发送更新：

```python
def subgraph_node(state):
    result = process(state)
    return Command(
        graph=Command.PARENT,
        update={"result": result},
        goto="next_in_parent"
    )
```

注意，`map_command()` 中对 `graph == Command.PARENT` 的处理会抛出 `InvalidUpdateError`。这是因为 `map_command()` 处理的是当前图层级的 Command。跨图的 Command 在执行引擎的更上层处理，由父图的 channel write 机制接收。

## Send 类详解

**源码路径**: `/tmp/langgraph-src/libs/langgraph/langgraph/types.py`

`Send` 是 Command.goto 中的核心构件，值得单独深入分析：

```python
class Send:
    """A message or packet to send to a specific node in the graph."""

    __slots__ = ("node", "arg")

    node: str
    arg: Any

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg

    def __hash__(self) -> int:
        return hash((self.node, self.arg))

    def __repr__(self) -> str:
        return f"Send(node={self.node!r}, arg={self.arg!r})"

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Send)
            and self.node == value.node
            and self.arg == value.arg
        )
```

`Send` 的设计特点：

1. **`__slots__`**：只有 `node` 和 `arg` 两个属性，使用 slots 优化内存
2. **可哈希**：实现了 `__hash__` 和 `__eq__`，可以用于集合和字典
3. **`/` 参数语法**：`__init__(self, /, node, arg)` 中的 `/` 表示 `node` 和 `arg` 只能作为位置参数传入

### Send 的 map-reduce 模式

`Send` 最经典的用途是实现 map-reduce：

```python
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from typing import Annotated
import operator

class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]

def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

builder = StateGraph(OverallState)
builder.add_node("generate_joke", lambda state: {
    "jokes": [f"Joke about {state['subject']}"]
})
builder.add_conditional_edges(START, continue_to_jokes)
builder.add_edge("generate_joke", END)
graph = builder.compile()

result = graph.invoke({"subjects": ["cats", "dogs"]})
# {'subjects': ['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}
```

每个 `Send` 创建一个独立的 push-style task，这些 task 在同一个 superstep 中并行执行。task 的输入是 Send 的 `arg`，而不是图的当前状态。

### Send 在 map_command 中的转换

```python
if isinstance(send, Send):
    yield (NULL_TASK_ID, TASKS, send)
```

`TASKS` 常量对应 `__pregel_tasks`，是 Pregel 的保留 channel。写入 TASKS 的 Send 对象会在 `prepare_next_tasks` 中被转换为 `PregelExecutableTask`，标记为 PUSH 类型（与 PULL 类型的边触发 task 区分）。

## 完整时间旅行工作流示例

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from typing import Annotated
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]
    step_count: int

def step_a(state):
    return {
        "messages": [f"step_a (count={state.get('step_count', 0)})"],
        "step_count": state.get("step_count", 0) + 1
    }

def step_b(state):
    return {
        "messages": [f"step_b (count={state['step_count']})"],
        "step_count": state["step_count"] + 1
    }

def step_c(state):
    return {
        "messages": [f"step_c (count={state['step_count']})"],
        "step_count": state["step_count"] + 1
    }

builder = StateGraph(State)
builder.add_node("a", step_a)
builder.add_node("b", step_b)
builder.add_node("c", step_c)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "timeline-1"}}

# --- 正常执行 ---
result = graph.invoke({"messages": ["start"], "step_count": 0}, config)

# --- 查看历史 ---
for snapshot in graph.get_state_history(config):
    step = snapshot.metadata.get("step", "?")
    print(f"Step {step}: next={snapshot.next}, "
          f"count={snapshot.values.get('step_count', '?')}")

# Step 3: next=(), count=3
# Step 2: next=('c',), count=2
# Step 1: next=('b',), count=1
# Step 0: next=('a',), count=0

# --- Fork：从 step_a 完成后的 checkpoint 重新执行 ---
for snapshot in graph.get_state_history(config):
    if snapshot.metadata.get("step") == 1:
        fork_config = snapshot.config
        break

# 从 step 1 开始，修改 step_count 为 100
for chunk in graph.stream(
    Command(update={"step_count": 100}),
    fork_config
):
    print(chunk)
# step_b 和 step_c 会以 step_count=100 为基础执行
```

## 时间旅行与 interrupt 的结合

时间旅行可以与 interrupt 结合使用——从 interrupt 状态的历史 checkpoint fork，提供不同的 resume 值：

```python
# 原始执行：interrupt 后 resume "approved"
for chunk in graph.stream({"data": "test"}, config):
    pass  # 遇到 interrupt

for chunk in graph.stream(Command(resume="approved"), config):
    pass  # 正常完成

# 时间旅行：回到 interrupt 点，这次 resume "rejected"
for snapshot in graph.get_state_history(config):
    if snapshot.interrupts:
        interrupt_config = snapshot.config
        break

for chunk in graph.stream(
    Command(resume="rejected"),
    interrupt_config
):
    pass  # 以 "rejected" 重新执行
```

## 时间旅行的限制与注意事项

1. **需要 checkpointer**：没有 checkpointer 就没有历史，无法时间旅行。
2. **状态可序列化**：checkpoint 需要序列化所有 channel 值。不可序列化的对象（如文件句柄、数据库连接）不能作为状态。
3. **副作用不可回滚**：时间旅行只回滚图的状态，不能撤销已经发生的外部副作用（API 调用、数据库写入等）。
4. **分支与主线独立**：fork 后的执行是新分支，不影响原始历史。
5. **子图独立存储**：子图的 checkpoint 在自己的 namespace 下，父图时间旅行不会自动回滚子图。
6. **`get_state_history` 的 `list()` 调用**：源码中用 `list()` 急切消费迭代器，避免长时间持有数据库游标，但会占用更多内存。

## GraphOutput 与 v2 API

在 v2 API 中，`invoke` 返回 `GraphOutput` 对象而非直接返回状态字典：

```python
# libs/langgraph/langgraph/types.py
@dataclass(frozen=True)
class GraphOutput(Generic[OutputT]):
    value: OutputT
    interrupts: tuple[Interrupt, ...] = ()
```

`GraphOutput` 同时包含输出值和 interrupt 信息，使得客户端可以区分正常完成和被中断的执行：

```python
result = graph.invoke(input, config, version="v2")
if result.interrupts:
    print("Graph was interrupted")
    for intr in result.interrupts:
        print(f"  Interrupt: {intr.value}")
else:
    print(f"Result: {result.value}")
```

为了向后兼容，`GraphOutput` 实现了 `__getitem__` 和 `__contains__`，但这些已被标记为 deprecated。

## 本章要点

1. **Command 四字段协同**：`update` 修改状态、`goto` 控制路由、`resume` 恢复中断、`graph` 指定目标图。它们可以自由组合使用。

2. **map_command 映射规则**：字符串 goto 映射为 `branch:to:{name}` channel write；`Send` 映射为 TASKS channel write；update 拆分为独立 channel writes。

3. **StateSnapshot 的完整信息**：包含 `values`（状态值）、`next`（下一步节点）、`config`（可用于恢复的配置）、`tasks`（任务列表，可能包含 error 和 interrupts）等。

4. **时间旅行的实现**：通过 `get_state_history()` 获取历史快照，使用快照的 `config` 恢复执行。重放模式（`is_replaying`）下会清除旧的 RESUME writes。

5. **Fork 是隐式的**：LangGraph 没有显式的 fork API。Fork 通过 "获取历史 config + invoke" 实现。新的执行会从该 checkpoint 继续，形成一条新的执行分支。

6. **update_state vs Command(update=...)**：前者是外部 API，直接操作 checkpoint；后者是图执行内部的控制流机制。

7. **Send 实现 map-reduce**：每个 Send 创建独立的 push-style task，支持并行执行。Send 是可哈希的值对象，通过 `__slots__` 优化内存。

8. **`_update_as_tuples()` 统一多种 update 格式**：dict、元组列表、Pydantic model、dataclass 和原始值都能正确处理。

9. **GraphOutput 统一了输出和 interrupt**：v2 API 中 `invoke` 返回 `GraphOutput(value=..., interrupts=...)`，客户端无需特殊处理 `__interrupt__` 键。

10. **时间旅行与 interrupt 可组合**：从 interrupt 状态的历史 checkpoint fork，提供不同 resume 值，探索不同的执行路径。
