# 第 5 章 节点、边与路由

在前几章中，我们学习了 StateGraph 的状态建模和 Channel 机制。现在是时候深入探讨图的核心构建单元——**节点（Node）** 和 **边（Edge）**——以及它们之间的路由控制方式了。

本章将从节点函数签名开始，逐步展开到类型注入、Managed Values、静态边与条件边、Send 动态路由、以及 Command 路由控制。我们将深入阅读源码，理解 LangGraph 是如何在编译期和运行期处理这些概念的。

---

## 5.1 节点函数签名

### 5.1.1 基本签名：state -> dict

在 StateGraph 中，节点的本质非常简单：它是一个接受当前状态、返回状态更新的函数。最基本的签名如下：

```python
def my_node(state: State) -> dict:
    return {"key": "value"}
```

这里 `state` 是整个图状态的快照（或其子集，如果指定了 `input_schema`），返回值是一个字典，表示需要更新的状态字段。

LangGraph 在源码中通过 Protocol 类精确地定义了所有合法的节点签名。让我们来看
`/tmp/langgraph-src/libs/langgraph/langgraph/graph/_node.py`：

```python
class _Node(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra) -> Any: ...


class _NodeWithConfig(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, config: RunnableConfig) -> Any: ...


class _NodeWithWriter(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, writer: StreamWriter) -> Any: ...


class _NodeWithStore(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, store: BaseStore) -> Any: ...
```

可以看到，LangGraph 定义了一系列 Protocol 类，每一个代表一种合法的节点函数签名。这些 Protocol 最终通过 `TypeAlias` 联合在一起，形成完整的 `StateNode` 类型：

```python
StateNode: TypeAlias = (
    _Node[NodeInputT]
    | _NodeWithConfig[NodeInputT]
    | _NodeWithWriter[NodeInputT]
    | _NodeWithStore[NodeInputT]
    | _NodeWithWriterStore[NodeInputT]
    | _NodeWithConfigWriter[NodeInputT]
    | _NodeWithConfigStore[NodeInputT]
    | _NodeWithConfigWriterStore[NodeInputT]
    | _NodeWithRuntime[NodeInputT, ContextT]
    | Runnable[NodeInputT, Any]
)
```

### 5.1.2 同步与异步均支持

LangGraph 的节点函数既可以是同步的，也可以是异步的。框架在内部通过 `coerce_to_runnable` 将普通函数包装为 `RunnableCallable`，它能同时处理同步和异步调用：

```python
# 同步节点
def sync_node(state: State) -> dict:
    result = call_some_api(state["query"])
    return {"result": result}

# 异步节点
async def async_node(state: State) -> dict:
    result = await async_call_some_api(state["query"])
    return {"result": result}
```

当你调用 `graph.invoke()` 时，同步节点直接执行；当你调用 `graph.ainvoke()` 时，异步节点会被原生地 `await`，而同步节点会被包装后在适当的上下文中执行。

### 5.1.3 Runnable 作为节点

除了普通函数之外，任何 LangChain `Runnable` 对象都可以作为节点使用。这在 `StateNode` 的类型定义中有明确体现：

```python
| Runnable[NodeInputT, Any]
```

这意味着你可以直接将 LLM、Chain 或其他 Runnable 作为节点添加到图中：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
builder.add_node("llm", llm)
```

---

## 5.2 类型注入

LangGraph 支持通过函数签名中的关键字参数自动注入多种运行时对象。这种机制让节点可以在需要时访问额外的上下文信息，而不必将所有内容都放入状态中。

### 5.2.1 config: RunnableConfig

`RunnableConfig` 是 LangChain 生态系统中的核心配置对象，包含了运行时的各种配置信息（如 `thread_id`、`callbacks` 等）。在节点函数中声明 `config` 参数即可自动注入：

```python
from langchain_core.runnables import RunnableConfig

def my_node(state: State, config: RunnableConfig) -> dict:
    thread_id = config["configurable"]["thread_id"]
    # 可以基于 thread_id 做不同处理
    return {"result": f"processed in thread {thread_id}"}
```

对应的 Protocol 定义为：

```python
class _NodeWithConfig(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, config: RunnableConfig) -> Any: ...
```

需要注意的是，`config` 可以作为位置参数（第二个参数）传入，而其他注入对象必须作为关键字参数传入。

### 5.2.2 store: BaseStore

`BaseStore` 提供了持久化的键值存储能力，可以跨线程、跨会话地共享信息。在节点中声明 `store` 关键字参数即可注入：

```python
from langgraph.store.base import BaseStore

def my_node(state: State, *, store: BaseStore) -> dict:
    # 从 store 中读取用户信息
    user_info = store.get(("users",), state["user_id"])
    if user_info:
        return {"greeting": f"Welcome back, {user_info.value['name']}!"}
    return {"greeting": "Welcome, new user!"}
```

Store 的注入通过 `_NodeWithStore` Protocol 声明：

```python
class _NodeWithStore(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, store: BaseStore) -> Any: ...
```

### 5.2.3 writer: StreamWriter

`StreamWriter` 是一个可调用对象，用于向 `stream_mode="custom"` 的流中写入自定义数据：

```python
from langgraph.types import StreamWriter

def my_node(state: State, *, writer: StreamWriter) -> dict:
    writer({"progress": "starting..."})
    result = do_some_work()
    writer({"progress": "done!"})
    return {"result": result}
```

`StreamWriter` 的类型定义非常简洁：

```python
StreamWriter = Callable[[Any], None]
```

它只接受一个参数并写入输出流。当不使用 `stream_mode="custom"` 时，`StreamWriter` 是一个空操作（no-op）。

### 5.2.4 runtime: Runtime

`Runtime` 是 LangGraph v0.6.0 引入的新机制，它将多种运行时资源（`context`、`store`、`stream_writer`、`previous`）封装在一个对象中。让我们查看其定义，位于 `/tmp/langgraph-src/libs/langgraph/langgraph/runtime.py`：

```python
@dataclass(**_DC_KWARGS)
class Runtime(Generic[ContextT]):
    context: ContextT = field(default=None)
    """Static context for the graph run, like user_id, db_conn, etc."""

    store: BaseStore | None = field(default=None)
    """Store for the graph run, enabling persistence and memory."""

    stream_writer: StreamWriter = field(default=_no_op_stream_writer)
    """Function that writes to the custom stream."""

    previous: Any = field(default=None)
    """The previous return value for the given thread."""
```

`Runtime` 的设计意图是统一所有运行时注入，让节点可以通过一个参数访问所有资源：

```python
from langgraph.runtime import Runtime

class Context(TypedDict):
    user_id: str

def my_node(state: State, *, runtime: Runtime[Context]) -> dict:
    user_id = runtime.context["user_id"]
    if runtime.store:
        data = runtime.store.get(("users",), user_id)
    runtime.stream_writer({"status": "processing"})
    return {"result": "done"}
```

对应的 Protocol：

```python
class _NodeWithRuntime(Protocol[NodeInputT_contra, ContextT]):
    def __call__(
        self, state: NodeInputT_contra, *, runtime: Runtime[ContextT]
    ) -> Any: ...
```

### 5.2.5 组合注入

以上注入类型可以自由组合。例如，同时需要 `config`、`writer` 和 `store` 的节点：

```python
class _NodeWithConfigWriterStore(Protocol[NodeInputT_contra]):
    def __call__(
        self,
        state: NodeInputT_contra,
        *,
        config: RunnableConfig,
        writer: StreamWriter,
        store: BaseStore,
    ) -> Any: ...
```

实际使用：

```python
def comprehensive_node(
    state: State,
    *,
    config: RunnableConfig,
    writer: StreamWriter,
    store: BaseStore,
) -> dict:
    thread_id = config["configurable"]["thread_id"]
    writer({"thread": thread_id, "status": "start"})

    cached = store.get(("cache",), state["key"])
    if cached:
        writer({"thread": thread_id, "status": "cache_hit"})
        return {"result": cached.value}

    result = compute_expensive_thing(state)
    store.put(("cache",), state["key"], {"value": result})
    writer({"thread": thread_id, "status": "done"})
    return {"result": result}
```

### 5.2.6 StateNodeSpec：节点的编译期表示

当你调用 `add_node` 时，LangGraph 会将节点函数包装为一个 `StateNodeSpec` 数据类。这是节点在编译前的内部表示：

```python
@dataclass(slots=True)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode[NodeInputT, ContextT]
    metadata: dict[str, Any] | None
    input_schema: type[NodeInputT]
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    ends: tuple[str, ...] | dict[str, str] | None = EMPTY_SEQ
    defer: bool = False
```

其中各字段的含义：

| 字段 | 说明 |
|------|------|
| `runnable` | 节点的实际执行逻辑，即用户传入的函数或 Runnable |
| `metadata` | 附加到节点的元数据，用于追踪和调试 |
| `input_schema` | 节点的输入 schema，默认为图的 state_schema |
| `retry_policy` | 重试策略 |
| `cache_policy` | 缓存策略 |
| `ends` | 节点可能路由到的目标（用于 Command 路由的渲染） |
| `defer` | 是否延迟执行到图运行即将结束时 |

---

## 5.3 Managed Values

### 5.3.1 什么是 Managed Values

Managed Values 是 LangGraph 中一种特殊的"虚拟状态字段"。与普通 Channel 不同，Managed Values 不被持久化到 checkpoint 中，而是在运行时动态计算。它们通过 `Annotated` 类型声明在 State 中，但由 `ManagedValue` 子类的 `get()` 方法在每次访问时实时生成。

Managed Values 的基类定义在 `/tmp/langgraph-src/libs/langgraph/langgraph/managed/base.py`：

```python
class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...
```

`ManagedValue` 是一个抽象基类，子类必须实现 `get` 静态方法。该方法接收 `PregelScratchpad`（Pregel 引擎的运行时草稿板），并返回计算后的值。

### 5.3.2 IsLastStep

`IsLastStep` 是最常用的 Managed Value 之一。它告诉节点当前是否处于图的最后一步（基于递归限制 `recursion_limit`）。定义在 `/tmp/langgraph-src/libs/langgraph/langgraph/managed/is_last_step.py`：

```python
class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1


IsLastStep = Annotated[bool, IsLastStepManager]
```

使用方法：

```python
from langgraph.managed.is_last_step import IsLastStep

class State(TypedDict):
    messages: list[str]
    is_last_step: IsLastStep

def my_node(state: State) -> dict:
    if state["is_last_step"]:
        return {"messages": ["Final step reached, wrapping up..."]}
    return {"messages": ["Still processing..."]}
```

关键要点：
- `IsLastStep` 本质是 `Annotated[bool, IsLastStepManager]`
- 它通过比较 `scratchpad.step`（当前步数）和 `scratchpad.stop`（最大步数，即 `recursion_limit`）来计算
- 不会被存储到 checkpoint 中——每次读取时实时计算

### 5.3.3 RemainingSteps

`RemainingSteps` 类似于 `IsLastStep`，但返回的是剩余步数而非布尔值：

```python
class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step


RemainingSteps = Annotated[int, RemainingStepsManager]
```

使用方法：

```python
from langgraph.managed.is_last_step import RemainingSteps

class State(TypedDict):
    messages: list[str]
    remaining_steps: RemainingSteps

def my_node(state: State) -> dict:
    remaining = state["remaining_steps"]
    if remaining <= 2:
        return {"messages": [f"Only {remaining} steps left, simplifying..."]}
    return {"messages": ["Proceeding with full analysis..."]}
```

### 5.3.4 Managed Values 的识别机制

LangGraph 是如何区分普通 Channel 字段和 Managed Values 的？答案在 `_get_channels` 函数中（位于 `/tmp/langgraph-src/libs/langgraph/langgraph/graph/state.py`）：

```python
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

这个函数将 State schema 的所有字段分为两类：
1. **BaseChannel 实例** — 普通的状态 Channel（如 `LastValue`、`BinaryOperatorAggregate`）
2. **ManagedValueSpec** — Managed Value 类型

区分的关键在于 `_is_field_managed_value` 函数，它检查 `Annotated` 类型的 metadata 是否是 `ManagedValue` 的子类：

```python
def _is_field_managed_value(name: str, typ: type[Any]) -> ManagedValueSpec | None:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1:
            decoration = get_origin(meta[-1]) or meta[-1]
            if is_managed_value(decoration):
                return decoration
    return None
```

### 5.3.5 Managed Values 的限制

Managed Values 有一个重要限制：**它们不能出现在 Input/Output schema 中**。这是因为 Managed Values 是运行时概念，不能作为图的外部接口。源码中有明确的校验：

```python
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
```

当注册 `input_schema` 或 `output_schema` 时，`allow_managed=False` 会阻止包含 Managed Values 的 schema。

---

## 5.4 静态边

### 5.4.1 add_edge：无条件跳转

最简单的边类型是**静态边（Static Edge）**——从一个节点无条件地跳转到另一个节点。使用 `add_edge` 方法添加：

```python
builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge("a", "b")       # a 执行完后，总是跳转到 b
builder.add_edge(START, "a")     # 图开始时，首先执行 a
builder.add_edge("b", END)       # b 执行完后，图结束
```

`add_edge` 的源码实现很直接（位于 `StateGraph` 类中）：

```python
def add_edge(self, start_key: str | list[str], end_key: str) -> Self:
    if isinstance(start_key, str):
        if start_key == END:
            raise ValueError("END cannot be a start node")
        if end_key == START:
            raise ValueError("START cannot be an end node")
        self.edges.add((start_key, end_key))
        return self
    # 多起点的情况（waiting edges）
    for start in start_key:
        if start == END:
            raise ValueError("END cannot be a start node")
        if start not in self.nodes:
            raise ValueError(f"Need to add_node `{start}` first")
    if end_key == START:
        raise ValueError("START cannot be an end node")
    self.waiting_edges.add((tuple(start_key), end_key))
    return self
```

关键逻辑：
- 单起点边：直接添加到 `self.edges` 集合
- 多起点边（`start_key` 是列表时）：添加到 `self.waiting_edges`，表示需要等待所有起点都完成后才执行终点节点

### 5.4.2 多起点等待边

当 `start_key` 是一个列表时，LangGraph 创建一种"等待边"（waiting edge），它会等待所有指定的起点节点都完成后，才触发终点节点。这在并行执行后汇聚的场景中非常有用：

```python
builder.add_edge(["fetch_data", "fetch_config"], "process")
```

在编译时，这种等待边会被转换为 `NamedBarrierValue` Channel，我们将在第 6 章详细讨论。

### 5.4.3 set_entry_point 与 set_finish_point

这两个方法是 `add_edge` 的语法糖：

```python
def set_entry_point(self, key: str) -> Self:
    return self.add_edge(START, key)

def set_finish_point(self, key: str) -> Self:
    return self.add_edge(key, END)
```

### 5.4.4 add_sequence：快速构建线性流

`add_sequence` 方法可以一次性添加一系列按顺序执行的节点：

```python
def add_sequence(
    self,
    nodes: Sequence[StateNode | tuple[str, StateNode]],
) -> Self:
    if len(nodes) < 1:
        raise ValueError("Sequence requires at least one node.")

    previous_name: str | None = None
    for node in nodes:
        if isinstance(node, tuple) and len(node) == 2:
            name, node = node
        else:
            name = _get_node_name(node)
        self.add_node(name, node)
        if previous_name is not None:
            self.add_edge(previous_name, name)
        previous_name = name
    return self
```

使用示例：

```python
builder.add_sequence([step_1, step_2, step_3])
# 等同于：
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
```

注意 `add_sequence` **不会**自动添加 START -> 第一个节点 和 最后一个节点 -> END 的边，你仍需手动设置。

---

## 5.5 条件边

### 5.5.1 add_conditional_edges 方法

静态边的路由在构建时就已确定，而**条件边（Conditional Edge）**允许在运行时根据状态动态决定下一个节点。这是 LangGraph 最强大的特性之一：

```python
def route_by_type(state: State) -> str:
    if state["type"] == "urgent":
        return "fast_path"
    return "normal_path"

builder.add_conditional_edges("classifier", route_by_type)
```

`add_conditional_edges` 的源码：

```python
def add_conditional_edges(
    self,
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]]
    | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
    | Runnable[Any, Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
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
    return self
```

方法做了三件事：
1. 将路由函数转换为 Runnable
2. 创建 `BranchSpec` 对象（条件边的内部表示）
3. 如果路由函数有类型注解的输入 schema，自动注册

### 5.5.2 BranchSpec：条件边的数据结构

`BranchSpec` 是条件边的核心数据结构，定义在 `/tmp/langgraph-src/libs/langgraph/langgraph/graph/_branch.py`：

```python
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None
    input_schema: type[Any] | None = None
```

三个字段：
- `path`: 路由函数（已转换为 Runnable）
- `ends`: 路由值到节点名的映射（可选）
- `input_schema`: 路由函数的输入类型（可选，用于推断）

### 5.5.3 path_map 参数

`path_map` 允许你将路由函数的返回值映射为实际的节点名称：

```python
# 方式 1：字典映射
builder.add_conditional_edges(
    "classifier",
    classify,
    path_map={"positive": "handle_positive", "negative": "handle_negative"}
)

# 方式 2：列表（返回值与节点名相同）
builder.add_conditional_edges(
    "classifier",
    classify,
    path_map=["handle_positive", "handle_negative"]
)

# 方式 3：省略 path_map，通过返回值的 Literal 类型推断
def classify(state: State) -> Literal["handle_positive", "handle_negative"]:
    ...
builder.add_conditional_edges("classifier", classify)
```

`BranchSpec.from_path` 方法处理了这三种情况：

```python
@classmethod
def from_path(
    cls,
    path: Runnable[Any, Hashable | list[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None,
    infer_schema: bool = False,
) -> BranchSpec:
    path_map_: dict[Hashable, str] | None = None
    try:
        if isinstance(path_map, dict):
            path_map_ = path_map.copy()
        elif isinstance(path_map, list):
            path_map_ = {name: name for name in path_map}
        else:
            # 尝试从函数返回类型推断
            func: Callable | None = None
            if isinstance(path, (RunnableCallable, RunnableLambda)):
                func = path.func or path.afunc
            if func is not None:
                if rtn_type := get_type_hints(func).get("return"):
                    if get_origin(rtn_type) is Literal:
                        path_map_ = {name: name for name in get_args(rtn_type)}
    except Exception:
        pass
    # ...
    return cls(path=path, ends=path_map_, input_schema=input_schema)
```

### 5.5.4 条件边的运行时执行

条件边在运行时的执行流程由 `BranchSpec._route` 和 `BranchSpec._finish` 方法控制：

```python
def _route(
    self,
    input: Any,
    config: RunnableConfig,
    *,
    reader: Callable[[RunnableConfig], Any] | None,
    writer: _Writer,
) -> Runnable:
    if reader:
        value = reader(config)
        if (
            isinstance(value, dict)
            and isinstance(input, dict)
            and self.input_schema is None
        ):
            value = {**input, **value}
    else:
        value = input
    result = self.path.invoke(value, config)
    return self._finish(writer, input, result, config)
```

执行流程：
1. 如果有 `reader`（读取最新状态的函数），先读取最新状态
2. 调用路由函数 `self.path.invoke(value, config)`
3. 调用 `_finish` 处理路由结果

`_finish` 方法负责将路由结果转换为实际的 Channel 写入操作：

```python
def _finish(
    self,
    writer: _Writer,
    input: Any,
    result: Any,
    config: RunnableConfig,
) -> Runnable | Any:
    if not isinstance(result, (list, tuple)):
        result = [result]
    if self.ends:
        destinations: Sequence[Send | str] = [
            r if isinstance(r, Send) else self.ends[r] for r in result
        ]
    else:
        destinations = cast(Sequence[Send | str], result)
    if any(dest is None or dest == START for dest in destinations):
        raise ValueError("Branch did not return a valid destination")
    if any(p.node == END for p in destinations if isinstance(p, Send)):
        raise InvalidUpdateError("Cannot send a packet to the END node")
    entries = writer(destinations, False)
    # ...
```

注意几个验证规则：
- 路由不能返回 `None` 或 `START`
- `Send` 不能发送到 `END` 节点
- 路由可以返回多个目标（列表/元组），实现并行分支

### 5.5.5 异步条件边

条件边同样支持异步路由函数。`BranchSpec._aroute` 是异步版本：

```python
async def _aroute(
    self,
    input: Any,
    config: RunnableConfig,
    *,
    reader: Callable[[RunnableConfig], Any] | None,
    writer: _Writer,
) -> Runnable:
    if reader:
        value = reader(config)
        if (
            isinstance(value, dict)
            and isinstance(input, dict)
            and self.input_schema is None
        ):
            value = {**input, **value}
    else:
        value = input
    result = await self.path.ainvoke(value, config)
    return self._finish(writer, input, result, config)
```

唯一的区别是路由函数使用 `await self.path.ainvoke(value, config)` 进行异步调用。`_finish` 逻辑完全相同。

### 5.5.6 条件入口边

通过 `set_conditional_entry_point`，可以在图的入口处使用条件路由：

```python
def set_conditional_entry_point(
    self,
    path: Callable[..., Hashable | Sequence[Hashable]]
    | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
    | Runnable[Any, Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
    return self.add_conditional_edges(START, path, path_map)
```

这实际上就是 `add_conditional_edges(START, ...)` 的语法糖。

---

## 5.6 Send：动态路由与 Map-Reduce 模式

### 5.6.1 Send 的定义

`Send` 是 LangGraph 中实现动态路由的核心类型，定义在 `/tmp/langgraph-src/libs/langgraph/langgraph/types.py`：

```python
class Send:
    """A message or packet to send to a specific node in the graph.

    The `Send` class is used within a `StateGraph`'s conditional edges to
    dynamically invoke a node with a custom state at the next step.

    Importantly, the sent state can differ from the core graph's state,
    allowing for flexible and dynamic workflow management.
    """

    __slots__ = ("node", "arg")

    node: str
    arg: Any

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg
```

`Send` 有两个属性：
- `node`: 目标节点的名称
- `arg`: 要传递给目标节点的参数（可以与图的主状态不同！）

### 5.6.2 Send 与普通路由的区别

普通条件边返回节点名称字符串，目标节点会接收当前图状态作为输入。而 `Send` 允许你：

1. **自定义输入数据**：`Send` 的 `arg` 参数可以是任何类型，不必与图状态一致
2. **多次调用同一节点**：可以创建多个指向同一节点的 `Send`，每个携带不同的输入
3. **实现 Map-Reduce 模式**：将一个任务拆分为多个并行子任务

### 5.6.3 Map-Reduce 模式

这是 `Send` 最经典的用法。以下是源码文档中的示例：

```python
from typing import Annotated
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph
import operator

class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]

def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

builder = StateGraph(OverallState)
builder.add_node(
    "generate_joke",
    lambda state: {"jokes": [f"Joke about {state['subject']}"]}
)
builder.add_conditional_edges(START, continue_to_jokes)
builder.add_edge("generate_joke", END)
graph = builder.compile()

result = graph.invoke({"subjects": ["cats", "dogs"]})
# {'subjects': ['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}
```

执行流程：

```
START
  |-- continue_to_jokes（条件边函数）
  |   |-- Send("generate_joke", {"subject": "cats"})
  |   +-- Send("generate_joke", {"subject": "dogs"})
  |-- generate_joke（并行实例 1：subject=cats）
  |   +-- return {"jokes": ["Joke about cats"]}
  +-- generate_joke（并行实例 2：subject=dogs）
      +-- return {"jokes": ["Joke about dogs"]}
END
  +-- jokes 通过 operator.add reducer 合并
```

### 5.6.4 Send 在条件边中的处理

在 `BranchSpec._finish` 方法中，`Send` 对象被特殊处理：

```python
destinations: Sequence[Send | str] = [
    r if isinstance(r, Send) else self.ends[r] for r in result
]
```

如果路由函数返回 `Send` 对象，它会被直接保留（不通过 `path_map` 映射）。然后在 `writer` 函数中，`Send` 对象会被写入 `TASKS` Channel，由 Pregel 引擎创建相应的 task。

### 5.6.5 Send 的限制

源码中有一个重要的验证规则：

```python
if any(p.node == END for p in destinations if isinstance(p, Send)):
    raise InvalidUpdateError("Cannot send a packet to the END node")
```

不能创建 `Send(END, ...)` — `END` 不是一个真正的节点，不能接收 Send。

### 5.6.6 Send 的相等性与哈希

`Send` 实现了 `__hash__` 和 `__eq__`，允许它被用在集合和字典中：

```python
def __hash__(self) -> int:
    return hash((self.node, self.arg))

def __eq__(self, value: object) -> bool:
    return (
        isinstance(value, Send)
        and self.node == value.node
        and self.arg == value.arg
    )
```

这使得 Pregel 引擎可以通过比较 Send 对象来判断是否需要创建新的 task。

---

## 5.7 Command：节点返回路由控制

### 5.7.1 Command 的定义

`Command` 是 LangGraph 中另一种重要的路由机制，它允许**节点自身**决定下一步的路由，而不需要在外部定义条件边。定义在 `/tmp/langgraph-src/libs/langgraph/langgraph/types.py`：

```python
@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    """One or more commands to update the graph's state and send messages to nodes.

    Args:
        graph: Graph to send the command to. Supported values are:
            - None: the current graph
            - Command.PARENT: closest parent graph
        update: Update to apply to the graph's state.
        resume: Value to resume execution with.
        goto: Can be one of the following:
            - Name of the node to navigate to next
            - Sequence of node names to navigate to next
            - Send object (to execute a node with the input provided)
            - Sequence of Send objects
    """

    graph: str | None = None
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()
```

### 5.7.2 Command 的四个字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `graph` | `str \| None` | 目标图。`None` 为当前图，`Command.PARENT` 为父图 |
| `update` | `Any \| None` | 要应用到状态的更新 |
| `resume` | `Any \| None` | 恢复中断执行的值 |
| `goto` | `Send \| Sequence[Send \| N] \| N` | 下一步要跳转的节点 |

### 5.7.3 Command 与条件边的对比

条件边将路由逻辑与节点逻辑分离——节点负责计算，条件边负责路由。而 `Command` 将两者合一：

```python
# 使用条件边的方式
def decide_node(state: State) -> dict:
    return {"analysis": analyze(state["data"])}

def route(state: State) -> str:
    if state["analysis"]["score"] > 0.8:
        return "fast_path"
    return "slow_path"

builder.add_conditional_edges("decide", route)

# 使用 Command 的方式
def decide_node(state: State) -> Command[Literal["fast_path", "slow_path"]]:
    analysis = analyze(state["data"])
    if analysis["score"] > 0.8:
        return Command(update={"analysis": analysis}, goto="fast_path")
    return Command(update={"analysis": analysis}, goto="slow_path")
```

Command 方式更加简洁，因为不需要额外的条件边函数，路由决策可以与节点逻辑紧密耦合。

### 5.7.4 Command 的状态更新提取

当节点返回 `Command` 时，LangGraph 需要从中提取状态更新。`_update_as_tuples` 方法负责此工作：

```python
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

支持多种 `update` 格式：
- `dict`: 转换为键值对列表
- `list[tuple[str, Any]]`: 直接使用
- Pydantic model / dataclass: 通过 `get_update_as_tuples` 提取
- 其他非 None 值: 作为 `__root__` Channel 的更新

### 5.7.5 Command 的路由处理

Command 的路由处理在 `_control_branch` 函数中（位于 `/tmp/langgraph-src/libs/langgraph/langgraph/graph/state.py`）：

```python
def _control_branch(value: Any) -> Sequence[tuple[str, Any]]:
    if isinstance(value, Send):
        return ((TASKS, value),)
    commands: list[Command] = []
    if isinstance(value, Command):
        commands.append(value)
    elif isinstance(value, (list, tuple)):
        for cmd in value:
            if isinstance(cmd, Command):
                commands.append(cmd)
    rtn: list[tuple[str, Any]] = []
    for command in commands:
        if command.graph == Command.PARENT:
            raise ParentCommand(command)
        goto_targets = (
            [command.goto] if isinstance(command.goto, (Send, str)) else command.goto
        )
        for go in goto_targets:
            if isinstance(go, Send):
                rtn.append((TASKS, go))
            elif isinstance(go, str) and go != END:
                rtn.append((_CHANNEL_BRANCH_TO.format(go), None))
    return rtn
```

这个函数将 Command 的 `goto` 字段转化为 Channel 写入：
- `Send` 对象 -> 写入 `TASKS` Channel
- 字符串节点名 -> 写入 `branch:to:{node}` Channel
- `Command.PARENT` -> 抛出 `ParentCommand` 异常，由父图处理

### 5.7.6 Command.PARENT：跨图路由

`Command.PARENT` 是一个特殊的常量，用于向父图发送命令。这在子图需要控制父图路由时非常有用：

```python
Command.PARENT: ClassVar[Literal["__parent__"]] = "__parent__"
```

使用示例：

```python
def subgraph_node(state: SubState) -> Command:
    if state["needs_escalation"]:
        return Command(
            graph=Command.PARENT,
            update={"escalated": True},
            goto="human_review"
        )
    return Command(update={"processed": True})
```

当 `_control_branch` 检测到 `command.graph == Command.PARENT` 时，会抛出 `ParentCommand` 异常。这个异常会被父图的 Pregel 引擎捕获并处理。

### 5.7.7 Command 的类型推断

当节点的返回类型注解为 `Command[Literal["node_a", "node_b"]]` 时，LangGraph 在 `add_node` 中会自动推断可能的路由目标：

```python
if (
    rtn_origin is Command
    and (rargs := get_args(rtn))
    and get_origin(rargs[0]) is Literal
    and (vals := get_args(rargs[0]))
):
    ends = vals
```

这种推断主要用于图的可视化渲染——让用户在图形界面上看到节点可能的路由方向，但不影响实际执行逻辑。

### 5.7.8 destinations 参数

如果类型推断不够用，`add_node` 方法还提供了 `destinations` 参数来显式声明路由目标：

```python
builder.add_node(
    "router",
    router_func,
    destinations={"fast": "fast_path", "slow": "slow_path"}
)
```

这同样仅用于图渲染，不影响执行。

### 5.7.9 Command 与 interrupt 的配合

Command 的 `resume` 字段用于恢复因 `interrupt()` 暂停的执行。这是 human-in-the-loop 工作流的核心机制：

```python
def human_review(state: State) -> dict:
    answer = interrupt("Please review this data")
    return {"review_result": answer}

# 在客户端恢复时：
graph.stream(Command(resume="approved"), config)
```

`interrupt` 函数的源码展示了其内部机制：

```python
def interrupt(value: Any) -> Any:
    conf = get_config()["configurable"]
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
        (Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),)
    )
```

执行流程：
1. 首次调用 `interrupt()` 时，抛出 `GraphInterrupt` 异常，图暂停
2. 客户端通过 `Command(resume=...)` 恢复执行
3. 节点从头重新执行，当再次遇到 `interrupt()` 时，返回之前提供的 resume 值
4. 如果节点中有多个 `interrupt()` 调用，通过 `idx`（计数器）按顺序匹配 resume 值

---

## 5.8 Overwrite：绕过 Reducer

在介绍路由之外，还有一个与节点返回值密切相关的类型——`Overwrite`。它允许节点绕过 Channel 的 reducer 函数，直接覆写值：

```python
@dataclass(slots=True)
class Overwrite:
    """Bypass a reducer and write the wrapped value directly
    to a BinaryOperatorAggregate channel."""

    value: Any
```

使用示例：

```python
from langgraph.types import Overwrite

class State(TypedDict):
    messages: Annotated[list, operator.add]

def reset_node(state: State) -> dict:
    # 正常返回会通过 operator.add 追加
    # 使用 Overwrite 可以直接替换整个列表
    return {"messages": Overwrite(value=["fresh start"])}
```

这在需要"重置"状态的场景中非常有用。

---

## 5.9 RetryPolicy 与 CachePolicy

### 5.9.1 RetryPolicy

节点可以配置重试策略，在遇到异常时自动重试：

```python
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    max_attempts: int = 3
    jitter: bool = True
    retry_on: (
        type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool]
    ) = default_retry_on
```

使用方式：

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "api_call",
    api_node,
    retry_policy=RetryPolicy(max_attempts=5, initial_interval=1.0)
)
```

### 5.9.2 CachePolicy

缓存策略允许节点的输出在相同输入下被缓存：

```python
@dataclass(**_DC_KWARGS)
class CachePolicy(Generic[KeyFuncT]):
    key_func: KeyFuncT = default_cache_key
    ttl: int | None = None
```

使用方式：

```python
from langgraph.types import CachePolicy

builder.add_node(
    "expensive_computation",
    compute_node,
    cache_policy=CachePolicy(ttl=3600)  # 缓存 1 小时
)
```

---

## 5.10 边的数据流总结

下面总结一下各种边类型在编译后的 Channel 写入方式：

| 边类型 | 源码方法 | Channel 操作 |
|--------|----------|-------------|
| 静态边 `A -> B` | `attach_edge` | A 的 writer 写入 `branch:to:B` |
| 等待边 `[A,B] -> C` | `attach_edge` | 创建 `NamedBarrierValue` Channel，A/B 各写入自己的名字 |
| 条件边 | `attach_branch` | 运行时 `_route`/`_aroute` 决定写入哪个 `branch:to:X` |
| Send | `_control_branch` | 写入 `TASKS` Channel |
| Command(goto=...) | `_control_branch` | 写入 `branch:to:X` 或 `TASKS` |
| Command(graph=PARENT) | `_control_branch` | 抛出 `ParentCommand` 异常 |

所有这些操作最终都转化为 Channel 写入，由 Pregel 引擎统一调度。`branch:to:{node}` 是 LangGraph 为每个节点自动创建的触发 Channel——当有值写入该 Channel 时，对应的节点就会在下一个 super step 中被触发执行。

---

## 5.11 实战示例：综合路由

让我们用一个完整的例子串联本章的概念：

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command, RetryPolicy
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps
from langgraph.runtime import Runtime
import operator

# State 定义
class State(TypedDict):
    query: str
    results: Annotated[list[str], operator.add]
    final_answer: str
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps

class Context(TypedDict):
    model_name: str

# 节点 1：分析查询，决定搜索策略
def analyzer(state: State, *, runtime: Runtime[Context]) -> Command[Literal["searcher", "direct_answer"]]:
    model = runtime.context["model_name"]
    if len(state["query"]) > 100:
        # 长查询：拆分为多个子查询并行搜索
        sub_queries = split_query(state["query"])
        return Command(
            goto=[Send("searcher", {"sub_query": q}) for q in sub_queries]
        )
    else:
        return Command(goto="direct_answer")

# 节点 2：搜索（可被 Send 多次并行调用）
def searcher(state: dict) -> dict:
    result = search(state["sub_query"])
    return {"results": [result]}

# 节点 3：直接回答
def direct_answer(state: State) -> dict:
    if state["is_last_step"]:
        return {"final_answer": "Time limit reached, returning partial answer"}
    return {"final_answer": answer(state["query"])}

# 节点 4：汇总结果
def summarizer(state: State) -> dict:
    return {"final_answer": summarize(state["results"])}

# 构建图
builder = StateGraph(state_schema=State, context_schema=Context)
builder.add_node(analyzer)
builder.add_node(searcher, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node(direct_answer)
builder.add_node(summarizer)

builder.add_edge(START, "analyzer")
builder.add_edge("searcher", "summarizer")
builder.add_edge("direct_answer", END)
builder.add_edge("summarizer", END)

graph = builder.compile()
result = graph.invoke(
    {"query": "Explain quantum computing"},
    context={"model_name": "gpt-4o"}
)
```

这个例子展示了：
- `Runtime` 注入获取 context
- `Command` 实现节点内路由
- `Send` 实现 Map-Reduce 并行搜索
- `IsLastStep` Managed Value 进行保护性检查
- `RetryPolicy` 为不稳定的搜索节点配置重试
- 静态边连接确定性路径

---

## 本章要点

1. **节点签名灵活**：LangGraph 通过 Protocol 类型系统支持 9 种以上的节点签名变体，从最简单的 `state -> dict` 到带有 `config`、`store`、`writer`、`runtime` 注入的复杂签名。同步和异步函数均可使用。

2. **类型注入机制**：`RunnableConfig`、`BaseStore`、`StreamWriter`、`Runtime` 可以通过关键字参数自动注入。其中 `Runtime` 是最新的统一方案，聚合了 context、store、stream_writer 和 previous。

3. **Managed Values 是虚拟字段**：`IsLastStep` 和 `RemainingSteps` 等 Managed Values 不被持久化，而是在运行时通过 `PregelScratchpad` 实时计算。它们通过 `Annotated` 类型元数据与 `ManagedValue` 子类关联。

4. **静态边 vs 条件边**：静态边（`add_edge`）在编译时确定路由；条件边（`add_conditional_edges`）在运行时通过路由函数动态决定。条件边内部表示为 `BranchSpec`，路由结果最终转化为 `branch:to:{node}` Channel 写入。

5. **Send 实现 Map-Reduce**：`Send` 允许条件边向同一节点发送多个不同的输入，实现并行处理。每个 `Send` 创建一个独立的 task，其结果通过 reducer 汇聚回主状态。

6. **Command 统一路由与更新**：`Command` 将状态更新（`update`）和路由控制（`goto`）合为一体，支持字符串目标、`Send` 对象、以及 `Command.PARENT` 跨图路由。它还可以通过 `resume` 字段恢复 `interrupt()` 暂停的执行。

7. **所有路由最终都是 Channel 写入**：无论是静态边、条件边、Send 还是 Command，最终都转化为 Channel 写入操作（`branch:to:{node}` 或 `TASKS`），由 Pregel 引擎统一调度。这种统一的抽象是 LangGraph 架构优雅之处。
