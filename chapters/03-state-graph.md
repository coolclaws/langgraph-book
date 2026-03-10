# 第 3 章 StateGraph：声明式图构建 API

在前两章中，我们了解了 LangGraph 的整体架构和 Pregel 执行引擎。本章将深入分析用户最常接触的 API 层——`StateGraph`。它是构建 LangGraph 应用的声明式入口，提供了 `add_node`、`add_edge`、`add_conditional_edges` 等方法，让开发者以直观的方式定义有状态的计算图。

> 源码文件：`libs/langgraph/langgraph/graph/state.py`（约 1752 行）

---

## 3.1 模块导出与入口组织

LangGraph 的 `graph` 包通过 `__init__.py` 提供了统一的导出接口：

```python
# libs/langgraph/langgraph/graph/__init__.py
from langgraph.constants import END, START
from langgraph.graph.message import MessageGraph, MessagesState, add_messages
from langgraph.graph.state import StateGraph

__all__ = (
    "END",
    "START",
    "StateGraph",
    "add_messages",
    "MessagesState",
    "MessageGraph",
)
```

这意味着用户只需要 `from langgraph.graph import StateGraph, START, END` 即可获得构建图所需的核心组件。这个简洁的导出列表背后，`StateGraph` 承载了绝大部分图构建逻辑，而 `START` 和 `END` 则是两个特殊的虚拟节点常量。

---

## 3.2 StateGraph vs MessageGraph：两种入口

### 3.2.1 StateGraph——通用状态图

`StateGraph` 是 LangGraph 的核心图构建器。它接受一个 state schema（通常是 `TypedDict`、Pydantic `BaseModel` 或 `dataclass`），并基于该 schema 自动创建底层的 channel 通道。

```python
# libs/langgraph/langgraph/graph/state.py, 第 115-184 行
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    """A graph whose nodes communicate by reading and writing to a shared state.

    The signature of each node is `State -> Partial<State>`.

    Each state key can optionally be annotated with a reducer function that
    will be used to aggregate the values of that key received from multiple nodes.
    The signature of a reducer function is `(Value, Value) -> Value`.
    """

    edges: set[tuple[str, str]]
    nodes: dict[str, StateNodeSpec[Any, ContextT]]
    branches: defaultdict[str, dict[str, BranchSpec]]
    channels: dict[str, BaseChannel]
    managed: dict[str, ManagedValueSpec]
    schemas: dict[type[Any], dict[str, BaseChannel | ManagedValueSpec]]
    waiting_edges: set[tuple[tuple[str, ...], str]]

    compiled: bool
    state_schema: type[StateT]
    context_schema: type[ContextT] | None
    input_schema: type[InputT]
    output_schema: type[OutputT]
```

`StateGraph` 使用了 Python 的 `Generic` 四类型参数：

| 类型参数 | 说明 |
|---------|------|
| `StateT` | 图的状态类型，核心数据结构 |
| `ContextT` | 运行时上下文（不可变，用于传递 `user_id`、数据库连接等） |
| `InputT` | 图的输入类型，默认等于 `StateT` |
| `OutputT` | 图的输出类型，默认等于 `StateT` |

这一设计允许用户在保持状态结构完整的同时，定义不同于状态的输入和输出接口。

### 3.2.2 MessageGraph——面向消息的简化入口（已废弃）

`MessageGraph` 是 `StateGraph` 的一个子类，专为消息列表场景设计。在 LangGraph 1.0 中它已被标记为 deprecated：

```python
# libs/langgraph/langgraph/graph/message.py, 第 247-304 行
@deprecated(
    "MessageGraph is deprecated in langgraph 1.0.0, to be removed in 2.0.0. "
    "Please use StateGraph with a `messages` key instead.",
    category=None,
)
class MessageGraph(StateGraph):
    """A StateGraph where every node receives a list of messages as input
    and returns one or more messages as output."""

    def __init__(self) -> None:
        warnings.warn(
            "MessageGraph is deprecated in LangGraph v1.0.0, to be removed in v2.0.0. "
            "Please use StateGraph with a `messages` key instead.",
            category=LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        super().__init__(Annotated[list[AnyMessage], add_messages])
```

`MessageGraph` 的全部实现只有一行关键代码：用 `Annotated[list[AnyMessage], add_messages]` 作为状态 schema 调用父类构造函数。这意味着整个状态只有一个 `__root__` 通道，其值是消息列表，使用 `add_messages` 作为 reducer。

### 3.2.3 推荐的替代方案：MessagesState

官方推荐使用 `MessagesState` 配合 `StateGraph`：

```python
# libs/langgraph/langgraph/graph/message.py, 第 307-308 行
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

使用方式：

```python
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState

# 可以直接使用 MessagesState
graph = StateGraph(MessagesState)

# 也可以继承它并添加自定义字段
class MyState(MessagesState):
    user_name: str
    score: int

graph = StateGraph(MyState)
```

这种方式既保留了消息管理的便利性，又允许用户扩展状态结构，比 `MessageGraph` 更加灵活。

### 3.2.4 add_messages reducer 的深入分析

`add_messages` 是 LangGraph 中最重要的内置 reducer 之一。它实现了消息列表的智能合并：

```python
# libs/langgraph/langgraph/graph/message.py, 第 60-244 行
@_add_messages_wrapper
def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Literal["langchain-openai"] | None = None,
) -> Messages:
    """Merges two lists of messages, updating existing messages by ID."""
```

核心合并逻辑如下：

1. **将输入强制转换为列表格式**
2. **为缺少 ID 的消息分配 UUID**
3. **检测 `REMOVE_ALL_MESSAGES` 标记**——如果遇到特殊的 `RemoveMessage(id="__remove_all__")`，则丢弃之前所有消息
4. **按 ID 合并**——如果新消息的 ID 与现有消息匹配，则替换；如果新消息是 `RemoveMessage`，则删除对应消息
5. **可选的 OpenAI 格式转换**——`format="langchain-openai"` 参数可自动格式化输出

```python
# 合并核心逻辑（简化）
merged = left.copy()
merged_by_id = {m.id: i for i, m in enumerate(merged)}
ids_to_remove = set()
for m in right:
    if (existing_idx := merged_by_id.get(m.id)) is not None:
        if isinstance(m, RemoveMessage):
            ids_to_remove.add(m.id)
        else:
            ids_to_remove.discard(m.id)
            merged[existing_idx] = m
    else:
        if isinstance(m, RemoveMessage):
            raise ValueError(
                f"Attempting to delete a message with an ID that "
                f"doesn't exist ('{m.id}')"
            )
        merged_by_id[m.id] = len(merged)
        merged.append(m)
merged = [m for m in merged if m.id not in ids_to_remove]
```

值得注意的是 `_add_messages_wrapper` 装饰器让 `add_messages` 支持柯里化：当不传参时返回 partial 函数，这使得 `add_messages(format="langchain-openai")` 可以直接用在 `Annotated` 注解中。

---

## 3.3 StateGraph 构造函数

### 3.3.1 `__init__` 方法签名

```python
# libs/langgraph/langgraph/graph/state.py, 第 200-252 行
def __init__(
    self,
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    *,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
    **kwargs: Unpack[DeprecatedKwargs],
) -> None:
```

参数说明：

| 参数 | 类型 | 说明 |
|------|------|------|
| `state_schema` | `type[StateT]` | 必填，图的状态 schema |
| `context_schema` | `type[ContextT] \| None` | 可选，运行时上下文 schema |
| `input_schema` | `type[InputT] \| None` | 可选，输入 schema（默认使用 state_schema） |
| `output_schema` | `type[OutputT] \| None` | 可选，输出 schema（默认使用 state_schema） |

### 3.3.2 初始化过程

构造函数初始化了图的所有核心数据结构：

```python
self.nodes = {}           # 节点名 -> StateNodeSpec
self.edges = set()        # (source, target) 边集合
self.branches = defaultdict(dict)  # 条件分支
self.schemas = {}         # schema 类型 -> channel 映射
self.channels = {}        # channel 名 -> BaseChannel 实例
self.managed = {}         # 受管理的值（如 SharedValue）
self.compiled = False     # 是否已编译
self.waiting_edges = set()  # 等待多个前驱完成的边

self.state_schema = state_schema
self.input_schema = cast(type[InputT], input_schema or state_schema)
self.output_schema = cast(type[OutputT], output_schema or state_schema)
self.context_schema = context_schema
```

随后调用 `_add_schema` 方法将 schema 解析为 channel：

```python
self._add_schema(self.state_schema)
self._add_schema(self.input_schema, allow_managed=False)
self._add_schema(self.output_schema, allow_managed=False)
```

注意 `input_schema` 和 `output_schema` 不允许包含 managed value（如 `SharedValue`），因为它们仅用于数据的输入输出，不参与状态管理。

### 3.3.3 `_add_schema` 内部机制

```python
# libs/langgraph/langgraph/graph/state.py, 第 260-290 行
def _add_schema(self, schema: type[Any], /, allow_managed: bool = True) -> None:
    if schema not in self.schemas:
        _warn_invalid_state_schema(schema)
        channels, managed, type_hints = _get_channels(schema)
        if managed and not allow_managed:
            names = ", ".join(managed)
            schema_name = getattr(schema, "__name__", "")
            raise ValueError(
                f"Invalid managed channels detected in {schema_name}: {names}."
                " Managed channels are not permitted in Input/Output schema."
            )
        self.schemas[schema] = {**channels, **managed}
        for key, channel in channels.items():
            if key in self.channels:
                if self.channels[key] != channel:
                    if isinstance(channel, LastValue):
                        pass  # LastValue 可以覆盖
                    else:
                        raise ValueError(
                            f"Channel '{key}' already exists with a different type"
                        )
            else:
                self.channels[key] = channel
```

`_get_channels` 是一个关键的工厂函数（将在第 4 章详细分析），它解析 Python 类型注解并创建对应的 channel 实例。

---

## 3.4 add_node：添加节点

### 3.4.1 方法重载

`add_node` 提供了四个 `@overload` 签名以支持多种调用方式：

```python
# 方式 1：直接传入函数，名称自动推断
builder.add_node(my_function)

# 方式 2：传入函数，指定 input_schema
builder.add_node(my_function, input_schema=MyInput)

# 方式 3：指定名称和函数
builder.add_node("my_node", my_function)

# 方式 4：指定名称、函数和 input_schema
builder.add_node("my_node", my_function, input_schema=MyInput)
```

### 3.4.2 核心实现

实际实现方法的签名：

```python
# libs/langgraph/langgraph/graph/state.py, 第 572-786 行
def add_node(
    self,
    node: str | StateNode[NodeInputT, ContextT],
    action: StateNode[NodeInputT, ContextT] | None = None,
    *,
    defer: bool = False,
    metadata: dict[str, Any] | None = None,
    input_schema: type[NodeInputT] | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    destinations: dict[str, str] | tuple[str, ...] | None = None,
    **kwargs: Unpack[DeprecatedKwargs],
) -> Self:
```

关键参数说明：

| 参数 | 说明 |
|------|------|
| `node` | 节点函数/Runnable，或者节点名称（字符串） |
| `action` | 当 `node` 为字符串时，这是实际的节点函数 |
| `defer` | 是否将节点延迟到运行结束前执行 |
| `metadata` | 节点元数据 |
| `input_schema` | 节点输入 schema（默认使用图的 state_schema） |
| `retry_policy` | 重试策略 |
| `cache_policy` | 缓存策略 |
| `destinations` | 节点可能路由到的目标（用于图渲染） |

### 3.4.3 节点名称解析

当第一个参数不是字符串时，方法会自动推断节点名称：

```python
if not isinstance(node, str):
    action = node
    if isinstance(action, Runnable):
        node = action.get_name()
    else:
        node = getattr(action, "__name__", action.__class__.__name__)
    if node is None:
        raise ValueError(
            "Node name must be provided if action is not a function"
        )
```

### 3.4.4 名称合法性校验

方法对节点名称执行严格校验：

```python
if node in self.nodes:
    raise ValueError(f"Node `{node}` already present.")
if node == END or node == START:
    raise ValueError(f"Node `{node}` is reserved.")

for character in (NS_SEP, NS_END):
    if character in node:
        raise ValueError(
            f"'{character}' is a reserved character and is not allowed "
            f"in the node names."
        )
```

`NS_SEP` 和 `NS_END` 是 LangGraph 内部用于命名空间分隔的保留字符，不允许出现在用户定义的节点名称中。

### 3.4.5 输入 schema 推断

`add_node` 会尝试从函数的类型注解中推断输入 schema：

```python
if (
    isfunction(action)
    or ismethod(action)
    or ismethod(getattr(action, "__call__", None))
) and (
    hints := get_type_hints(getattr(action, "__call__"))
    or get_type_hints(action)
):
    if input_schema is None:
        first_parameter_name = next(
            iter(
                inspect.signature(
                    cast(FunctionType, action)
                ).parameters.keys()
            )
        )
        if input_hint := hints.get(first_parameter_name):
            if isinstance(input_hint, type) and get_type_hints(input_hint):
                inferred_input_schema = input_hint
```

这段逻辑检查函数的第一个参数的类型注解。如果它是一个拥有类型注解的类（通常是 `TypedDict` 或 `BaseModel`），则将其作为推断的输入 schema。

### 3.4.6 Command 返回类型检测

方法还会检查函数的返回类型注解，以自动提取 `Command` 类型中的目标节点信息：

```python
if rtn := hints.get("return"):
    rtn_origin = get_origin(rtn)
    if rtn_origin is Union:
        rtn_args = get_args(rtn)
        for arg in rtn_args:
            arg_origin = get_origin(arg)
            if arg_origin is Command:
                rtn = arg
                rtn_origin = arg_origin
                break

    if (
        rtn_origin is Command
        and (rargs := get_args(rtn))
        and get_origin(rargs[0]) is Literal
        and (vals := get_args(rargs[0]))
    ):
        ends = vals
```

如果函数的返回类型是 `Command[Literal["node_a", "node_b"]]`，LangGraph 会提取 `("node_a", "node_b")` 作为该节点的目标列表，用于图的可视化渲染。

### 3.4.7 StateNodeSpec 创建

最后，节点信息被包装为 `StateNodeSpec` 并存储：

```python
if input_schema is not None:
    self.nodes[node] = StateNodeSpec[NodeInputT, ContextT](
        coerce_to_runnable(action, name=node, trace=False),
        metadata,
        input_schema=input_schema,
        retry_policy=retry_policy,
        cache_policy=cache_policy,
        ends=ends,
        defer=defer,
    )
```

`coerce_to_runnable` 负责将普通函数转换为 LangChain 的 `Runnable` 对象，使其能被统一调度。

---

## 3.5 add_edge：添加边

### 3.5.1 方法签名

```python
# libs/langgraph/langgraph/graph/state.py, 第 788-840 行
def add_edge(self, start_key: str | list[str], end_key: str) -> Self:
    """Add a directed edge from the start node (or list of start nodes)
    to the end node."""
```

### 3.5.2 单源边

对于单个起始节点的情况：

```python
if isinstance(start_key, str):
    if start_key == END:
        raise ValueError("END cannot be a start node")
    if end_key == START:
        raise ValueError("START cannot be an end node")
    self.edges.add((start_key, end_key))
    return self
```

边被简单地存储为 `(source, target)` 元组。

### 3.5.3 多源等待边（Fan-in）

当传入节点列表时，形成一个 fan-in 等待边——只有所有起始节点都完成后，才会触发目标节点：

```python
for start in start_key:
    if start == END:
        raise ValueError("END cannot be a start node")
    if start not in self.nodes:
        raise ValueError(f"Need to add_node `{start}` first")
if end_key == START:
    raise ValueError("START cannot be an end node")
if end_key != END and end_key not in self.nodes:
    raise ValueError(f"Need to add_node `{end_key}` first")

self.waiting_edges.add((tuple(start_key), end_key))
```

等待边存储在 `waiting_edges` 集合中，编译时会被转换为 `NamedBarrierValue` channel。

### 3.5.4 使用示例

```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    x: int

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)

# 普通边
builder.add_edge(START, "a")
builder.add_edge("a", "b")

# Fan-in：等待 a 和 b 都完成后执行 c
builder.add_edge(["a", "b"], "c")
builder.add_edge("c", END)
```

---

## 3.6 add_conditional_edges：条件路由

### 3.6.1 方法签名

```python
# libs/langgraph/langgraph/graph/state.py, 第 842-890 行
def add_conditional_edges(
    self,
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]]
    | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
    | Runnable[Any, Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
    """Add a conditional edge from the starting node to any number of
    destination nodes."""
```

### 3.6.2 参数解析

| 参数 | 说明 |
|------|------|
| `source` | 条件分支的起始节点 |
| `path` | 路由函数，接收 state，返回目标节点名或列表 |
| `path_map` | 可选的路径映射表，将 path 的返回值映射为节点名 |

### 3.6.3 内部实现

```python
# find a name for the condition
path = coerce_to_runnable(path, name=None, trace=True)
name = path.name or "condition"
# validate the condition
if name in self.branches[source]:
    raise ValueError(
        f"Branch with name `{path.name}` already exists for node `{source}`"
    )
# save it
self.branches[source][name] = BranchSpec.from_path(path, path_map, True)
if schema := self.branches[source][name].input_schema:
    self._add_schema(schema)
```

条件边被存储为 `BranchSpec` 对象，在编译时会被转化为 Pregel 运行时的分支逻辑。`path` 函数的返回值支持：

- 返回单个字符串（节点名）
- 返回字符串列表（并行执行多个节点）
- 返回 `END` 常量（终止图执行）
- 返回 `Send` 对象（发送特定数据到目标节点）

### 3.6.4 使用示例

```python
def route(state: State) -> Literal["tool_node", "__end__"]:
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END

builder.add_conditional_edges("agent", route)

# 或者使用 path_map
def route_with_map(state: State) -> str:
    if state["score"] > 0.8:
        return "high"
    return "low"

builder.add_conditional_edges(
    "evaluator",
    route_with_map,
    path_map={"high": "approve", "low": "retry"}
)
```

---

## 3.7 辅助路由方法

### 3.7.1 set_entry_point

```python
# libs/langgraph/langgraph/graph/state.py, 第 939-950 行
def set_entry_point(self, key: str) -> Self:
    """Specifies the first node to be called in the graph.
    Equivalent to calling `add_edge(START, key)`."""
    return self.add_edge(START, key)
```

### 3.7.2 set_finish_point

```python
# libs/langgraph/langgraph/graph/state.py, 第 976-987 行
def set_finish_point(self, key: str) -> Self:
    """Marks a node as a finish point of the graph.
    If the graph reaches this node, it will cease execution."""
    return self.add_edge(key, END)
```

### 3.7.3 set_conditional_entry_point

```python
# libs/langgraph/langgraph/graph/state.py, 第 952-974 行
def set_conditional_entry_point(
    self,
    path: Callable[..., Hashable | Sequence[Hashable]]
    | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
    | Runnable[Any, Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
    """Sets a conditional entry point in the graph."""
    return self.add_conditional_edges(START, path, path_map)
```

这三个方法都是语法糖，分别等价于 `add_edge(START, key)`、`add_edge(key, END)` 和 `add_conditional_edges(START, path, path_map)`。

### 3.7.4 add_sequence

```python
# libs/langgraph/langgraph/graph/state.py, 第 892-937 行
def add_sequence(
    self,
    nodes: Sequence[
        StateNode[NodeInputT, ContextT]
        | tuple[str, StateNode[NodeInputT, ContextT]]
    ],
) -> Self:
    """Add a sequence of nodes that will be executed in the provided order."""
    if len(nodes) < 1:
        raise ValueError("Sequence requires at least one node.")

    previous_name: str | None = None
    for node in nodes:
        if isinstance(node, tuple) and len(node) == 2:
            name, node = node
        else:
            name = _get_node_name(node)

        if name in self.nodes:
            raise ValueError(
                f"Node names must be unique: node with the name '{name}' "
                f"already exists."
            )

        self.add_node(name, node)
        if previous_name is not None:
            self.add_edge(previous_name, name)

        previous_name = name

    return self
```

`add_sequence` 是一个便利方法，接受一个节点列表并按顺序用边连接它们。它简化了线性流水线的构建。

---

## 3.8 START 与 END 特殊节点

### 3.8.1 概念说明

`START` 和 `END` 是 LangGraph 定义的两个特殊虚拟节点常量。它们不是真正的计算节点，而是图的入口和出口标记：

- **`START`**：表示图的入口点。从 `START` 出发的边决定了图接收到输入后第一个执行的节点。
- **`END`**：表示图的终止点。到达 `END` 的边意味着该执行路径结束。

### 3.8.2 编译时的处理

在 `compile()` 方法中，`START` 被注册为一个特殊的 `PregelNode`：

```python
# libs/langgraph/langgraph/graph/state.py, 第 1300-1306 行
if key == START:
    self.nodes[key] = PregelNode(
        tags=[TAG_HIDDEN],
        triggers=[START],
        channels=START,
        writers=[ChannelWrite(write_entries)],
    )
```

`START` 节点的特殊之处：

1. 使用 `TAG_HIDDEN` 标记，不会出现在流式输出中
2. 由 `START` channel 触发（一个 `EphemeralValue` channel）
3. 其唯一作用是将用户输入写入状态 channel

### 3.8.3 START channel 的创建

```python
# libs/langgraph/langgraph/graph/state.py, 第 1148 行
channels={
    **self.channels,
    **self.managed,
    START: EphemeralValue(self.input_schema),
},
```

`START` 对应一个 `EphemeralValue` channel，接收用户输入后立即触发图的第一个节点，然后自动清空。

### 3.8.4 END 的处理

`END` 没有对应的节点或 channel。当一个边的目标是 `END` 时，编译器不会创建任何触发机制——这意味着执行路径自然终止。在 `attach_edge` 方法中：

```python
# libs/langgraph/langgraph/graph/state.py, 第 1339-1347 行
def attach_edge(self, starts: str | Sequence[str], end: str) -> None:
    if isinstance(starts, str):
        if end != END:
            self.nodes[starts].writers.append(
                ChannelWrite(
                    (ChannelWriteEntry(_CHANNEL_BRANCH_TO.format(end), None),)
                )
            )
```

只有当 `end != END` 时才会添加写入器。到达 `END` 的执行路径不会触发任何后续节点。

---

## 3.9 validate：图验证

在编译之前，`validate` 方法会对图结构进行完整性检查：

```python
# libs/langgraph/langgraph/graph/state.py, 第 989-1036 行
def validate(self, interrupt: Sequence[str] | None = None) -> Self:
    # assemble sources
    all_sources = {src for src, _ in self._all_edges}
    for start, branches in self.branches.items():
        all_sources.add(start)
    for name, spec in self.nodes.items():
        if spec.ends:
            all_sources.add(name)

    # validate sources
    for source in all_sources:
        if source not in self.nodes and source != START:
            raise ValueError(
                f"Found edge starting at unknown node '{source}'"
            )

    if START not in all_sources:
        raise ValueError(
            "Graph must have an entrypoint: add at least one edge "
            "from START to another node"
        )
```

验证内容包括：

1. **入口点检查**：必须有从 `START` 出发的边
2. **源节点存在性**：所有边的起始节点必须已注册
3. **目标节点存在性**：所有边的目标节点必须已注册（或者是 `END`）
4. **中断节点存在性**：`interrupt_before` 和 `interrupt_after` 中指定的节点必须存在

---

## 3.10 compile：从 Builder 到 Executable

### 3.10.1 方法签名

```python
# libs/langgraph/langgraph/graph/state.py, 第 1038-1048 行
def compile(
    self,
    checkpointer: Checkpointer = None,
    *,
    cache: BaseCache | None = None,
    store: BaseStore | None = None,
    interrupt_before: All | list[str] | None = None,
    interrupt_after: All | list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
```

### 3.10.2 编译流程

编译过程是从声明式 `StateGraph` 到可执行 `CompiledStateGraph` 的转换。主要步骤如下：

**第一步：序列化白名单构建**（如果启用了严格 msgpack 模式）

```python
if _serde.STRICT_MSGPACK_ENABLED:
    schema_types = [self.state_schema, self.input_schema, self.output_schema]
    # ...收集所有 schema 类型...
    serde_allowlist = _serde.build_serde_allowlist(
        schemas=schema_types, channels=self.channels
    )
```

**第二步：验证图结构**

```python
self.validate(
    interrupt=(
        (interrupt_before if interrupt_before != "*" else [])
        + interrupt_after if interrupt_after != "*" else []
    )
)
```

**第三步：确定输出 channel**

```python
output_channels = (
    "__root__"
    if len(self.schemas[self.output_schema]) == 1
    and "__root__" in self.schemas[self.output_schema]
    else [
        key
        for key, val in self.schemas[self.output_schema].items()
        if not is_managed_value(val)
    ]
)
```

如果输出 schema 只有一个 `__root__` key（例如 `MessageGraph` 的场景），则直接使用 `"__root__"` 字符串；否则使用 key 列表。

**第四步：创建 CompiledStateGraph**

```python
compiled = CompiledStateGraph[StateT, ContextT, InputT, OutputT](
    builder=self,
    schema_to_mapper={},
    context_schema=self.context_schema,
    nodes={},
    channels={
        **self.channels,
        **self.managed,
        START: EphemeralValue(self.input_schema),
    },
    input_channels=START,
    stream_mode="updates",
    output_channels=output_channels,
    stream_channels=stream_channels,
    checkpointer=checkpointer,
    interrupt_before_nodes=interrupt_before,
    interrupt_after_nodes=interrupt_after,
    auto_validate=False,
    debug=debug,
    store=store,
    cache=cache,
    name=name or "LangGraph",
)
```

**第五步：附加节点和边**

```python
compiled.attach_node(START, None)
for key, node in self.nodes.items():
    compiled.attach_node(key, node)

for start, end in self.edges:
    compiled.attach_edge(start, end)

for starts, end in self.waiting_edges:
    compiled.attach_edge(starts, end)

for start, branches in self.branches.items():
    for name, branch in branches.items():
        compiled.attach_branch(start, name, branch)
```

---

## 3.11 CompiledStateGraph：编译后的图

### 3.11.1 类定义

```python
# libs/langgraph/langgraph/graph/state.py, 第 1196-1214 行
class CompiledStateGraph(
    Pregel[StateT, ContextT, InputT, OutputT],
    Generic[StateT, ContextT, InputT, OutputT],
):
    builder: StateGraph[StateT, ContextT, InputT, OutputT]
    schema_to_mapper: dict[type[Any], Callable[[Any], Any] | None]
    _output_mapper: Callable[[Any], Any] | None
    _state_mapper: Callable[[Any], Any] | None
```

`CompiledStateGraph` 继承自 `Pregel`，是实际可执行的图对象。它持有对原始 `StateGraph` builder 的引用。

### 3.11.2 attach_node 详解

`attach_node` 是编译过程中最复杂的方法之一。它将 `StateNodeSpec` 转换为 Pregel 运行时的 `PregelNode`：

```python
# libs/langgraph/langgraph/graph/state.py, 第 1236-1337 行
def attach_node(self, key: str, node: StateNodeSpec[Any, ContextT] | None) -> None:
```

对于普通节点（非 `START`），核心转换逻辑如下：

1. **创建 branch channel**——为每个节点创建一个 `branch:to:{node_name}` 格式的触发 channel
2. **确定输入映射**——根据节点的 input_schema 决定读取哪些 channel
3. **创建 PregelNode**——包含触发器、输入 channel、输出写入器

```python
branch_channel = _CHANNEL_BRANCH_TO.format(key)
self.channels[branch_channel] = (
    LastValueAfterFinish(Any) if node.defer
    else EphemeralValue(Any, guard=False)
)
self.nodes[key] = PregelNode(
    triggers=[branch_channel],
    channels=("__root__" if is_single_input else input_channels),
    mapper=mapper,
    writers=[ChannelWrite(write_entries)],
    metadata=node.metadata,
    retry_policy=node.retry_policy,
    cache_policy=node.cache_policy,
    bound=node.runnable,
)
```

`defer=True` 的节点使用 `LastValueAfterFinish` channel 作为触发器。这种 channel 只有在 Pregel 运行的 `finish()` 阶段才会变为可用，从而实现延迟执行。

### 3.11.3 attach_edge 详解

```python
# libs/langgraph/langgraph/graph/state.py, 第 1339-1363 行
def attach_edge(self, starts: str | Sequence[str], end: str) -> None:
    if isinstance(starts, str):
        if end != END:
            self.nodes[starts].writers.append(
                ChannelWrite(
                    (ChannelWriteEntry(
                        _CHANNEL_BRANCH_TO.format(end), None
                    ),)
                )
            )
    elif end != END:
        channel_name = f"join:{'+'.join(starts)}:{end}"
        if self.builder.nodes[end].defer:
            self.channels[channel_name] = NamedBarrierValueAfterFinish(
                str, set(starts)
            )
        else:
            self.channels[channel_name] = NamedBarrierValue(str, set(starts))
        self.nodes[end].triggers.append(channel_name)
        for start in starts:
            self.nodes[start].writers.append(
                ChannelWrite((ChannelWriteEntry(channel_name, start),))
            )
```

这里展示了 LangGraph 如何将声明式的边定义转化为 Pregel 的 channel 机制：

- **单源边**：在源节点的 writers 中添加一个写入目标节点 `branch:to` channel 的动作
- **多源等待边**（fan-in）：创建一个 `NamedBarrierValue` channel，每个源节点完成后写入自己的名称，只有所有源都写入后目标节点才被触发

### 3.11.4 状态类型映射（mapper）

`_pick_mapper` 函数决定是否需要将 channel 读取的 dict 转换为 schema 类实例：

```python
# libs/langgraph/langgraph/graph/state.py, 第 1520-1527 行
def _pick_mapper(
    state_keys: Sequence[str], schema: type[Any]
) -> Callable[[Any], Any] | None:
    if state_keys == ["__root__"]:
        return None
    if isclass(schema) and (issubclass(schema, BaseModel) or is_dataclass(schema)):
        return partial(_coerce_state, schema)
    return None

def _coerce_state(schema: type[_S], input: dict[str, Any]) -> _S:
    return schema(**input)
```

- 对于 `__root__` 类型的 schema，不需要映射
- 对于 Pydantic `BaseModel` 和 `dataclass`，需要将 dict 转换为实例
- 对于 `TypedDict`，不需要映射（它本身就是 dict）

---

## 3.12 图的内部数据流

为了全面理解 `StateGraph` 的工作原理，我们来追踪一个简单图从构建到执行的完整数据流。

### 3.12.1 构建阶段

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    count: int

builder = StateGraph(State)
builder.add_node("agent", agent_fn)
builder.add_node("tool", tool_fn)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tool", "agent")
```

构建后，`builder` 的内部状态：

```
channels = {
    "messages": BinaryOperatorAggregate(list, add_messages),
    "count": LastValue(int),
}
nodes = {
    "agent": StateNodeSpec(runnable=agent_fn, ...),
    "tool": StateNodeSpec(runnable=tool_fn, ...),
}
edges = {(START, "agent"), ("tool", "agent")}
branches = {"agent": {"should_continue": BranchSpec(...)}}
```

### 3.12.2 编译阶段

```python
graph = builder.compile()
```

编译后，`graph` 的内部状态新增：

```
channels += {
    START: EphemeralValue(State),
    "branch:to:agent": EphemeralValue(Any, guard=False),
    "branch:to:tool": EphemeralValue(Any, guard=False),
}
nodes = {
    START: PregelNode(triggers=[START], writers=[...]),
    "agent": PregelNode(triggers=["branch:to:agent"], writers=[...]),
    "tool": PregelNode(triggers=["branch:to:tool"], writers=[...]),
}
```

### 3.12.3 执行阶段

当调用 `graph.invoke({"messages": [("user", "hello")]})` 时：

1. 输入写入 `START` channel
2. `START` channel 更新触发 START PregelNode
3. START 节点将输入解析后写入 `messages` 和 `count` channel
4. START 节点还写入 `branch:to:agent` channel（因为有 START -> agent 边）
5. `branch:to:agent` 更新触发 agent PregelNode
6. agent 节点读取所有 state channel，执行 `agent_fn`
7. agent 节点的输出写入 state channel
8. `should_continue` 分支函数被执行，决定下一步
9. 如果返回 "tool"，则写入 `branch:to:tool`，触发 tool 节点
10. tool 节点执行后写入 `branch:to:agent`，循环继续
11. 如果 `should_continue` 返回 END，不写入任何 branch channel，执行结束

---

## 3.13 _get_channels 与 _get_channel：Schema 到 Channel 的转换

### 3.13.1 _get_channels

```python
# libs/langgraph/langgraph/graph/state.py, 第 1603-1623 行
def _get_channels(
    schema: type[dict],
) -> tuple[dict[str, BaseChannel], dict[str, ManagedValueSpec], dict[str, Any]]:
    if not hasattr(schema, "__annotations__"):
        return (
            {"__root__": _get_channel("__root__", schema, allow_managed=False)},
            {},
            {},
        )

    type_hints = get_type_hints(schema, include_extras=True)
    all_keys = {
        name: _get_channel(name, typ)
        for name, typ in type_hints.items()
        if name != "__slots__"
    }
    return (
        {k: v for k, v in all_keys.items() if isinstance(v, BaseChannel)},
        {k: v for k, v in all_keys.items() if is_managed_value(v)},
        type_hints,
    )
```

如果 schema 没有 `__annotations__`（如直接传入 `Annotated[list, add_messages]`），则创建一个 `__root__` channel。否则，遍历所有类型注解，为每个字段创建对应的 channel。

### 3.13.2 _get_channel

```python
# libs/langgraph/langgraph/graph/state.py, 第 1638-1661 行
def _get_channel(
    name: str, annotation: Any, *, allow_managed: bool = True
) -> BaseChannel | ManagedValueSpec:
    # Strip out Required and NotRequired wrappers
    if hasattr(annotation, "__origin__") and annotation.__origin__ in (
        Required, NotRequired,
    ):
        annotation = annotation.__args__[0]
    if manager := _is_field_managed_value(name, annotation):
        if allow_managed:
            return manager
        else:
            raise ValueError(f"This {annotation} not allowed in this position")
    elif channel := _is_field_channel(annotation):
        channel.key = name
        return channel
    elif channel := _is_field_binop(annotation):
        channel.key = name
        return channel

    fallback: LastValue = LastValue(annotation)
    fallback.key = name
    return fallback
```

Channel 创建的优先级链：

1. **Managed Value 检测** -- 如果 `Annotated` 元数据中包含 managed value 类型
2. **显式 Channel 检测** (`_is_field_channel`) -- 如果元数据中包含 `BaseChannel` 实例或子类
3. **二元运算符检测** (`_is_field_binop`) -- 如果元数据中包含 reducer 函数
4. **默认 LastValue** -- 没有任何注解时，使用 `LastValue` channel

### 3.13.3 _is_field_binop 详解

```python
# libs/langgraph/langgraph/graph/state.py, 第 1678-1696 行
def _is_field_binop(typ: type[Any]) -> BinaryOperatorAggregate | None:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and callable(meta[-1]):
            sig = signature(meta[-1])
            params = list(sig.parameters.values())
            if (
                sum(
                    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    for p in params
                )
                == 2
            ):
                return BinaryOperatorAggregate(typ, meta[-1])
            else:
                raise ValueError(
                    f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
                )
    return None
```

这个函数检查 `Annotated` 类型的元数据中最后一个元素是否是一个接受两个位置参数的 callable。如果是，则将其视为 reducer 函数，创建 `BinaryOperatorAggregate` channel。

---

## 3.14 push_message：手动消息推送

`message.py` 还提供了一个 `push_message` 函数，允许在节点执行过程中手动向消息流写入数据：

```python
# libs/langgraph/langgraph/graph/message.py, 第 327-372 行
def push_message(
    message: MessageLikeRepresentation | BaseMessageChunk,
    *,
    state_key: str | None = "messages",
) -> AnyMessage:
    """Write a message manually to the `messages` / `messages-tuple` stream mode.
    Will automatically write to the channel specified in the `state_key`
    unless `state_key` is `None`."""
```

这对于流式输出中间结果特别有用——可以在节点执行过程中逐步推送消息给客户端。

---

## 3.15 方法链式调用

`StateGraph` 的所有构建方法都返回 `Self`，支持链式调用：

```python
graph = (
    StateGraph(State)
    .add_node("agent", agent_fn)
    .add_node("tool", tool_fn)
    .add_edge(START, "agent")
    .add_conditional_edges("agent", route)
    .add_edge("tool", "agent")
    .compile(checkpointer=MemorySaver())
)
```

这种设计模式使得图的构建代码更加简洁和流畅。

---

## 3.16 完整的图构建示例

下面是一个将本章所有概念串联起来的完整示例：

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, add_messages
from langgraph.checkpoint.memory import InMemorySaver


# 1. 定义状态（使用 Annotated 指定 reducer）
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int


# 2. 定义节点函数
def agent(state: AgentState) -> dict:
    # 调用 LLM，返回状态更新
    return {
        "messages": [("assistant", "I'm thinking...")],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def tool(state: AgentState) -> dict:
    return {"messages": [("tool", "Here are the results.")]}


# 3. 定义路由函数
def should_continue(state: AgentState) -> Literal["tool", "__end__"]:
    if state["iteration_count"] < 3:
        return "tool"
    return END


# 4. 构建图
builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tool", tool)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tool", "agent")

# 5. 编译
graph = builder.compile(checkpointer=InMemorySaver())

# 6. 执行
result = graph.invoke(
    {"messages": [("user", "Hello!")]},
    config={"configurable": {"thread_id": "1"}}
)
```

---

## 本章要点

1. **StateGraph 是声明式构建器**：它不直接执行图，而是收集节点、边和分支的定义，在 `compile()` 时转化为可执行的 `CompiledStateGraph`（继承自 Pregel）。

2. **MessageGraph 已废弃**：LangGraph 1.0 推荐使用 `StateGraph(MessagesState)` 替代 `MessageGraph`，后者将在 2.0 中移除。

3. **四种添加节点的方式**：`add_node` 支持自动名称推断、显式命名、指定 input_schema 等多种重载签名，同时支持 `defer`、`retry_policy`、`cache_policy` 等高级选项。

4. **三种边类型**：
   - `add_edge(A, B)` —— 无条件边
   - `add_edge([A, B], C)` —— Fan-in 等待边
   - `add_conditional_edges(A, fn)` —— 条件路由边

5. **START 和 END 的实现**：`START` 对应一个 `EphemeralValue` channel，负责将用户输入注入图中；`END` 是一个逻辑标记，到达 `END` 的路径不会触发后续节点。

6. **Schema 到 Channel 的自动转换**：`_get_channels` 和 `_get_channel` 函数解析 Python 类型注解，自动为每个状态字段创建合适的 channel（`LastValue`、`BinaryOperatorAggregate` 等）。

7. **编译过程**：`compile()` 将声明式的图定义转化为 Pregel 运行时结构——每个节点变为 `PregelNode`，每条边变为 channel 写入操作，等待边变为 `NamedBarrierValue` channel。

8. **方法链式调用**：所有构建方法返回 `Self`，支持 fluent API 风格的图构建。

9. **add_messages reducer**：实现了基于 ID 的消息智能合并，支持追加、替换和删除操作，是 LangGraph 聊天应用的核心组件。
