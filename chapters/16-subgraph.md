# 第 16 章 子图与命名空间隔离

在构建复杂的 Agent 系统时，将逻辑拆分为多个子图是一种自然的架构选择。LangGraph 通过命名空间（namespace）机制实现了父子图之间的状态隔离与通信。本章将深入分析子图的添加方式、NS_SEP / NS_END 命名空间分隔符、父子图状态共享与隔离机制、跨图 Send 以及 stream_mode 传播。

## 子图的添加：add_node(subgraph.compile())

在 LangGraph 中，子图的添加非常直观 -- 将一个编译后的图作为节点添加到父图中：

```python
from langgraph.graph import StateGraph, START, END

# 定义子图
class SubState(TypedDict):
    sub_input: str
    sub_output: str

sub_builder = StateGraph(SubState)
sub_builder.add_node("process", process_fn)
sub_builder.add_edge(START, "process")
sub_builder.add_edge("process", END)
sub_graph = sub_builder.compile()

# 将子图添加为父图的节点
class ParentState(TypedDict):
    input: str
    result: str

parent_builder = StateGraph(ParentState)
parent_builder.add_node("sub", sub_graph)
parent_builder.add_edge(START, "sub")
parent_builder.add_edge("sub", END)
parent_graph = parent_builder.compile(checkpointer=InMemorySaver())
```

当子图被添加为节点时，`StateGraph` 的 `add_node()` 方法对节点名中的保留字符进行检查：

```python
# libs/langgraph/langgraph/graph/state.py
for character in (NS_SEP, NS_END):
    if character in node:
        raise ValueError(
            f"'{character}' is a reserved character and is not allowed in the node names."
        )
```

这确保了节点名不会与命名空间分隔符冲突。

## 命名空间：NS_SEP / NS_END 分隔符

LangGraph 使用两个特殊字符构建层级化的命名空间：

```python
# libs/langgraph/langgraph/_internal/_constants.py
NS_SEP = sys.intern("|")
# for checkpoint_ns, separates each level (ie. graph|subgraph|subsubgraph)
NS_END = sys.intern(":")
# for checkpoint_ns, for each level, separates the namespace from the task_id
```

命名空间的完整格式为：

```
{node_name}{NS_END}{task_id}{NS_SEP}{child_node_name}{NS_END}{child_task_id}
```

例如，一个三层嵌套的图可能产生如下 checkpoint_ns：

```
outer_node:task-abc|inner_node:task-def|leaf_node:task-ghi
```

在 `_prepare_state_snapshot()` 中可以看到命名空间的组装方式：

```python
# libs/langgraph/langgraph/pregel/main.py
task_ns = f"{task.name}{NS_END}{task.id}"
if parent_ns:
    task_ns = f"{parent_ns}{NS_SEP}{task_ns}"
```

这种设计确保了每个子图的每次执行都有唯一的命名空间标识，即使同一个子图被并行执行多次也不会冲突。

## 父子图状态共享与隔离

### 状态隔离

默认情况下，父图和子图拥有各自独立的状态空间。子图的 State schema 可以与父图完全不同。当子图作为节点执行时：

1. 父图将当前状态（或 Send 的 arg）传递给子图作为输入
2. 子图在自己的状态空间中执行
3. 子图的输出被映射回父图的状态更新

### Checkpointer 继承

子图的 checkpointer 行为通过 `Checkpointer` 类型控制：

```python
# libs/langgraph/langgraph/types.py
Checkpointer = None | bool | BaseCheckpointSaver
"""Type of the checkpointer to use for a subgraph.

- `True` enables persistent checkpointing for this subgraph.
- `False` disables checkpointing, even if the parent graph has a checkpointer.
- `None` inherits checkpointer from the parent graph.
"""
```

在编译子图时可以指定 checkpointer 策略：

```python
# 继承父图的 checkpointer（默认行为）
sub_graph = sub_builder.compile()  # checkpointer=None

# 显式启用独立的 checkpointing
sub_graph = sub_builder.compile(checkpointer=True)

# 禁用 checkpointing
sub_graph = sub_builder.compile(checkpointer=False)
```

当 checkpointer 为 `None` 时，子图通过 `CONFIG_KEY_CHECKPOINTER` 从父图的 config 中继承 checkpointer：

```python
# libs/langgraph/langgraph/_internal/_constants.py
CONFIG_KEY_CHECKPOINTER = sys.intern("__pregel_checkpointer")
# holds a `BaseCheckpointSaver` passed from parent graph to child graphs
```

### 状态共享的几种模式

**模式一：通过相同的 State key 共享**

如果父子图的 State 有同名字段，LangGraph 会自动在边界处进行映射：

```python
class ParentState(TypedDict):
    messages: Annotated[list, operator.add]
    context: str

class ChildState(TypedDict):
    messages: Annotated[list, operator.add]
    internal_data: str

# messages 字段会自动在父子图间同步
```

**模式二：通过 Command 跨图通信**

子图可以通过 `Command(graph=Command.PARENT)` 直接更新父图的状态：

```python
def child_node(state):
    result = heavy_computation(state)
    return Command(
        graph=Command.PARENT,
        update={"result": result}
    )
```

## get_subgraphs：子图枚举

`get_subgraphs()` 方法允许遍历图的所有子图，支持按命名空间过滤和递归查找：

```python
# libs/langgraph/langgraph/pregel/main.py
def get_subgraphs(
    self, *, namespace: str | None = None, recurse: bool = False
) -> Iterator[tuple[str, PregelProtocol]]:
    for name, node in self.nodes.items():
        if namespace is not None:
            if not namespace.startswith(name):
                continue

        graph = node.subgraphs[0] if node.subgraphs else None

        if graph:
            if name == namespace:
                yield name, graph
                return
            if namespace is None:
                yield name, graph
            if recurse and isinstance(graph, Pregel):
                if namespace is not None:
                    namespace = namespace[len(name) + 1 :]
                yield from (
                    (f"{name}{NS_SEP}{n}", s)
                    for n, s in graph.get_subgraphs(
                        namespace=namespace, recurse=recurse
                    )
                )
```

递归模式下，子图的命名空间通过 `NS_SEP` 拼接，形成 `parent|child|grandchild` 格式。这与 `get_state()` 和 `get_state_history()` 中的子图路由机制配合：

```python
# libs/langgraph/langgraph/pregel/main.py (get_state)
if (
    checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
    recast = recast_checkpoint_ns(checkpoint_ns)
    for _, pregel in self.get_subgraphs(namespace=recast, recurse=True):
        return pregel.get_state(
            patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
            subgraphs=subgraphs,
        )
    else:
        raise ValueError(f"Subgraph {recast} not found")
```

这意味着你可以直接获取子图的状态，只需在 config 中指定正确的 `checkpoint_ns`。

## 跨图 Send

`Send` 对象主要用于在条件边中动态路由。在子图场景下，`Send` 的目标节点必须属于当前图层级。不能直接向子图内部的节点发送 `Send`。

然而，通过 `Command` 的 `goto` 字段结合 `Send`，可以实现灵活的跨层通信：

```python
# 在条件边中使用 Send 实现 map-reduce
def fan_out(state):
    return [
        Send("sub_graph_node", {"item": item})
        for item in state["items"]
    ]

builder.add_conditional_edges(START, fan_out)
```

每个 `Send` 会创建一个独立的 task，拥有自己的 task_id。这些 task 的命名空间格式为 `{node_name}{NS_END}{task_id}`，确保并行执行时互不干扰。

在 `map_command()` 中，Send 被写入 TASKS channel：

```python
# libs/langgraph/langgraph/pregel/_io.py
for send in sends:
    if isinstance(send, Send):
        yield (NULL_TASK_ID, TASKS, send)
    elif isinstance(send, str):
        yield (NULL_TASK_ID, f"branch:to:{send}", START)
```

TASKS channel 是 LangGraph 的内部机制，对应 `__pregel_tasks` 常量：

```python
# libs/langgraph/langgraph/_internal/_constants.py
TASKS = sys.intern("__pregel_tasks")
# for Send objects returned by nodes/edges, corresponds to PUSH below
PUSH = sys.intern("__pregel_push")
# denotes push-style tasks, ie. those created by Send objects
```

## 子图的 stream_mode 传播

当父图配置了 `stream_mode` 时，子图的流式输出会自动通过 `CONFIG_KEY_STREAM` 传播给父图：

```python
# libs/langgraph/langgraph/_internal/_constants.py
CONFIG_KEY_STREAM = sys.intern("__pregel_stream")
# holds a `StreamProtocol` passed from parent graph to child graphs
```

子图产生的 stream events 会带上命名空间前缀。在 stream output 的数据结构中，`ns` 字段是一个元组，记录了完整的命名空间路径：

```python
# libs/langgraph/langgraph/types.py
class ValuesStreamPart(TypedDict, Generic[OutputT]):
    type: Literal["values"]
    ns: tuple[str, ...]
    data: OutputT
    interrupts: tuple[Interrupt, ...]
```

例如，对于一个两层嵌套的图，子图的 stream event 可能看起来像：

```python
{
    "type": "updates",
    "ns": ("sub_graph_node",),
    "data": {"process": {"sub_output": "result"}}
}
```

更深层嵌套的子图会有更长的 ns 元组：

```python
{
    "type": "updates",
    "ns": ("level1_node", "level2_node"),
    "data": {"leaf": {"output": "deep_result"}}
}
```

消费端可以根据 `ns` 过滤感兴趣的子图事件：

```python
async for event in graph.astream(input, config, stream_mode="updates"):
    if event["ns"] == ():
        # 根图事件
        print("Root:", event["data"])
    elif event["ns"] == ("sub_graph_node",):
        # 子图事件
        print("Sub:", event["data"])
```

## 子图设计的最佳实践

### 何时使用子图

1. **逻辑封装**：当一组节点形成一个独立的业务单元时，将它们封装为子图可以提高代码的可维护性。
2. **状态隔离**：当子流程需要不同的状态 schema 时，子图提供了天然的隔离。
3. **复用**：同一个子图可以在多个父图中复用。
4. **团队协作**：不同团队可以独立开发和测试各自的子图。

### 注意事项

1. **checkpointer 策略**：默认继承父图的 checkpointer。如果子图不需要持久化，显式设置 `checkpointer=False` 可以减少存储开销。
2. **节点名冲突**：子图的节点名不需要与父图的节点名不同，因为命名空间机制会自动隔离。但需要避免在节点名中使用 `|` 和 `:` 这两个保留字符。
3. **Interrupt 传播**：子图中的 `interrupt()` 会正确传播到父图，通过命名空间机制保持 interrupt ID 的唯一性。

## is_nested 标志的影响

在 `PregelLoop` 中，`is_nested` 布尔标志标识当前图是否作为子图运行。这个标志影响多个关键行为。

### GraphInterrupt 传播

```python
# libs/langgraph/langgraph/pregel/_loop.py
suppress = isinstance(exc_value, GraphInterrupt) and not self.is_nested
```

- 根图（`is_nested=False`）：抑制 `GraphInterrupt`，通过 stream 输出 interrupt 信息
- 子图（`is_nested=True`）：不抑制，让异常向上传播到父图

这意味着子图中的 `interrupt()` 调用最终会被根图捕获和处理。中间的每一层子图只是透传异常。

### is_resuming 判断

```python
# libs/langgraph/langgraph/pregel/_loop.py
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

对于子图，`is_resuming` 主要通过父图设置的 `CONFIG_KEY_RESUMING` 标志来判断。子图的输入不是 `None` 或 `Command`（而是由父图节点产生的值），所以不能通过输入类型推断是否在 resume。

### durability="exit" 的持久化条件

```python
if self.durability == "exit" and (
    not self.is_nested
    or exc_value is not None
    or all(NS_END not in part for part in self.checkpoint_ns)
):
    self._put_checkpoint(self.checkpoint_metadata)
    self._put_pending_writes()
```

在 `durability="exit"` 模式下：
- 根图总是在退出时持久化
- 子图只在有异常或使用 `checkpointer=True`（独立 namespace）时持久化

`all(NS_END not in part for part in self.checkpoint_ns)` 检查 namespace 中是否没有 `:` 分隔符——如果没有，说明使用的是 `checkpointer=True` 模式的独立 namespace。

## CONFIG_KEY_RESUMING：子图恢复协调

```python
# libs/langgraph/langgraph/_internal/_constants.py
CONFIG_KEY_RESUMING = sys.intern("__pregel_resuming")
# holds a boolean indicating if subgraphs should resume from a previous checkpoint
```

当父图从 checkpoint 恢复时，子图不会直接接收 `Command(resume=...)` 输入。父图通过在 config 中设置 `CONFIG_KEY_RESUMING=True` 来通知子图：

1. 父图 resume 时设置 `CONFIG_KEY_RESUMING=True` 到子图的 config
2. 子图在 `_first()` 中检测到这个标志
3. 子图从自己的 checkpointer 加载之前的 checkpoint 状态
4. 子图的 `interrupt()` 调用使用之前缓存的 resume 值

这实现了跨层级的 resume 协调，确保整个图层级结构能够一致地恢复。

## CONFIG_KEY_REPLAY_STATE：重放状态追踪

```python
# libs/langgraph/langgraph/_internal/_constants.py
CONFIG_KEY_REPLAY_STATE = sys.intern("__pregel_replay_state")
# holds a ReplayState tracking the parent checkpoint_id upper bound
# and which subgraph namespaces have already loaded their pre-replay checkpoint
```

在时间旅行场景中，这个配置项追踪：
- 父图 checkpoint_id 的上界
- 哪些子图 namespace 已经加载了重放前的 checkpoint

这确保在从历史 checkpoint 重新执行时，子图也能正确回到对应的历史状态。

## 保留常量与系统安全

```python
# libs/langgraph/langgraph/_internal/_constants.py
RESERVED = {
    _TAG_HIDDEN,
    INPUT, INTERRUPT, RESUME, ERROR, NO_WRITES,
    CONFIG_KEY_SEND, CONFIG_KEY_READ, CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_STREAM, CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_RESUMING, CONFIG_KEY_REPLAY_STATE,
    CONFIG_KEY_TASK_ID, CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_ID, CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_RESUME_MAP,
    PUSH, PULL, NS_SEP, NS_END, CONF,
}
```

所有保留常量通过 `sys.intern()` 进行字符串驻留。驻留后的字符串在比较时可以使用 `is` 而非 `==`，提高性能。这些保留 key 不能被用户的 channel 名称使用，保证了系统内部通信的完整性。

`CONFIG_KEY_CHECKPOINT_MAP` 在子图恢复中尤为重要：

```python
CONFIG_KEY_CHECKPOINT_MAP = sys.intern("checkpoint_map")
# holds a mapping of checkpoint_ns -> checkpoint_id for parent graphs
```

它记录了各个 namespace 对应的 checkpoint_id，使得恢复时每个子图都能准确找到自己的 checkpoint。

## 完整子图示例

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from typing import Annotated
from typing_extensions import TypedDict
import operator

# --- 子图：处理单个任务 ---
class TaskState(TypedDict):
    task_name: str
    task_result: str

def process_task(state: TaskState):
    approval = interrupt(f"请审批任务: {state['task_name']}")
    if approval == "approved":
        return {"task_result": f"{state['task_name']} 已完成"}
    return {"task_result": f"{state['task_name']} 被拒绝"}

task_builder = StateGraph(TaskState)
task_builder.add_node("process", process_task)
task_builder.add_edge(START, "process")
task_builder.add_edge("process", END)
task_graph = task_builder.compile()

# --- 父图：编排任务 ---
class WorkflowState(TypedDict):
    tasks: list[str]
    results: Annotated[list[str], operator.add]

def dispatch(state: WorkflowState):
    if state["tasks"]:
        return {"task_name": state["tasks"][0]}
    return {}

workflow_builder = StateGraph(WorkflowState)
workflow_builder.add_node("dispatch", dispatch)
workflow_builder.add_node("task_processor", task_graph)
workflow_builder.add_edge(START, "dispatch")
workflow_builder.add_edge("dispatch", "task_processor")
workflow_builder.add_edge("task_processor", END)

checkpointer = InMemorySaver()
workflow = workflow_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "workflow-1"}}

# 执行时子图中的 interrupt 会传播到父图
for chunk in workflow.stream(
    {"tasks": ["部署服务"], "results": []},
    config
):
    print(chunk)

# 恢复子图的 interrupt
for chunk in workflow.stream(
    Command(resume="approved"),
    config
):
    print(chunk)
```

## 子图状态查询

### 通过 checkpoint_ns 查询子图状态

```python
sub_state = graph.get_state(
    {
        "configurable": {
            "thread_id": "my-thread",
            "checkpoint_ns": "sub_node"
        }
    }
)
```

### 递归获取子图状态

使用 `subgraphs=True` 可以在 `StateSnapshot` 中递归包含子图状态：

```python
state = graph.get_state(config, subgraphs=True)

for task in state.tasks:
    if task.state is not None:
        # task.state 是子图的 StateSnapshot 或 RunnableConfig
        if isinstance(task.state, StateSnapshot):
            print(f"Subgraph {task.name}: {task.state.values}")
```

在 `PregelTask` 中，`state` 字段可以是：

```python
# libs/langgraph/langgraph/types.py
class PregelTask(NamedTuple):
    id: str
    name: str
    path: tuple[str | int | tuple, ...]
    error: Exception | None = None
    interrupts: tuple[Interrupt, ...] = ()
    state: None | RunnableConfig | StateSnapshot = None
    result: Any | None = None
```

- `None`：不是子图节点
- `RunnableConfig`：子图的 config（可用于进一步查询 `get_state(task.state)`）
- `StateSnapshot`：子图的完整状态快照（当 `subgraphs=True` 时展开）

## 子图设计最佳实践

### 何时使用子图

1. **逻辑封装**：当一组节点形成一个独立的业务单元时，封装为子图提高可维护性
2. **状态隔离**：子流程需要不同的状态 schema 时，子图提供天然隔离
3. **模块复用**：同一个子图可以在多个父图中使用
4. **独立开发**：不同团队可以独立开发和测试各自的子图

### 何时避免子图

1. **简单流程**：只有几个节点的简单图，子图增加不必要的复杂度
2. **频繁跨图通信**：大量 `Command.PARENT` 使用导致代码难以理解
3. **性能敏感**：子图引入额外的 checkpoint 序列化开销

### 节点名的限制

节点名不能包含 `|`（NS_SEP）和 `:`（NS_END）这两个保留字符：

```python
# libs/langgraph/langgraph/graph/state.py
for character in (NS_SEP, NS_END):
    if character in node:
        raise ValueError(
            f"'{character}' is a reserved character and is not allowed in the node names."
        )
```

子图的节点名可以与父图相同，因为命名空间机制会自动隔离。

### Checkpointer 策略建议

```python
# 策略 1：子图继承父图 checkpointer（默认推荐）
# 子图共享父图的 checkpointer，但在独立 namespace 下
sub_graph = sub_builder.compile()  # checkpointer=None

# 策略 2：子图独立启用
# 子图使用 checkpointer=True，获得完全独立的 checkpoint 管理
sub_graph = sub_builder.compile(checkpointer=True)

# 策略 3：子图禁用 checkpointer
# 适合纯计算型子图，不需要持久化
sub_graph = sub_builder.compile(checkpointer=False)
```

选择策略时的考量：
- 默认策略（`None`）适用于大多数场景，子图自动继承父图的 checkpointer，checkpoint 存储在独立 namespace 下
- `True` 适用于需要完全独立的 checkpoint 历史的子图
- `False` 适用于纯计算型子图，减少不必要的序列化开销

## 本章要点

1. **命名空间格式**：`{node}{NS_END}{task_id}`（即 `node:task_id`）表示一层，多层用 `NS_SEP`（即 `|`）分隔。这种层级结构保证了每次执行的唯一性。

2. **状态隔离是默认行为**：父子图有独立的状态空间和 channel 系统。状态共享通过 State key 映射或 `Command(graph=Command.PARENT)` 实现。

3. **Checkpointer 的三种模式**：`None`（继承父图）、`True`（独立 checkpointing）、`False`（禁用）。默认 `None` 在大多数场景足够。

4. **get_subgraphs 的递归枚举**：支持按命名空间前缀过滤和递归查找，命名空间路径通过 `NS_SEP` 拼接。

5. **Send 创建并行 task**：每个 Send 产生独立的 task_id 和命名空间，支持同一子图的并行执行。

6. **Stream 事件带命名空间**：子图的 stream events 通过 `ns` 元组标识来源层级，支持消费端的精确过滤。

7. **is_nested 影响异常处理**：子图中的 `GraphInterrupt` 不被抑制，而是向上传播到根图统一处理。

8. **CONFIG_KEY_RESUMING 协调跨层级恢复**：父图 resume 时通过 config 标志通知子图，子图据此从自己的 checkpoint 恢复。

9. **保留常量通过 sys.intern() 驻留**：所有系统内部 key 使用字符串驻留优化比较性能，且不可被用户 channel 名覆盖。

10. **recast_checkpoint_ns 去除 task_id**：在查询子图实例时只需节点名称链，`recast_checkpoint_ns` 移除每层的 `:task_id` 部分。
