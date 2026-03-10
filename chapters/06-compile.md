# 第 6 章 编译：从声明图到可执行 Pregel

前几章中，我们逐一剖析了 State、Channel、节点与边的声明式 API。但 `StateGraph` 本身并不能运行——它只是一棵**声明树**。调用 `compile()` 之后，声明树才被转化为可执行的 `CompiledStateGraph`（继承自 `Pregel`）。本章将完整追踪这一编译过程，揭示从用户的 `builder.compile()` 到最终 `Pregel` 实例之间发生的每一步。

涉及的核心源码文件：

- `/tmp/langgraph-src/libs/langgraph/langgraph/graph/state.py` — `StateGraph.compile()`、`CompiledStateGraph`、`attach_node`、`attach_edge`、`attach_branch`
- `/tmp/langgraph-src/libs/langgraph/langgraph/pregel/_validate.py` — `validate_graph()` 验证函数
- `/tmp/langgraph-src/libs/langgraph/langgraph/pregel/_read.py` — `PregelNode`、`ChannelRead`

---

## 6.1 编译的全景视图

### 6.1.1 为什么需要编译

StateGraph 是一个面向用户的声明式构建器（Builder Pattern），它提供了 `add_node`、`add_edge`、`add_conditional_edges` 等友好的 API。但 Pregel 执行引擎需要的是另一种数据结构——由 `PregelNode`、`Channel`、`ChannelWrite` 和 `ChannelRead` 组成的低级别执行图。

编译过程的职责就是**桥接这两个世界**：

```
用户视角（声明式）          编译           引擎视角（执行式）
┌─────────────────┐    ──────────>    ┌─────────────────────┐
│ StateGraph      │                   │ CompiledStateGraph   │
│  .nodes         │                   │  .nodes: PregelNode  │
│  .edges         │                   │  .channels: Channel  │
│  .branches      │                   │  .input_channels     │
│  .channels      │                   │  .output_channels    │
│  .managed       │                   │  .stream_channels    │
└─────────────────┘                   └─────────────────────┘
```

### 6.1.2 compile() 方法签名

`compile()` 方法定义在 `StateGraph` 类中（`/tmp/langgraph-src/libs/langgraph/langgraph/graph/state.py`），它的完整签名为：

```python
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

参数说明：

| 参数 | 类型 | 说明 |
|------|------|------|
| `checkpointer` | `Checkpointer` | checkpoint 存储器，`None`/`True`/`False`/`BaseCheckpointSaver` |
| `cache` | `BaseCache \| None` | 节点缓存后端 |
| `store` | `BaseStore \| None` | 键值存储后端 |
| `interrupt_before` | `All \| list[str] \| None` | 在哪些节点前中断 |
| `interrupt_after` | `All \| list[str] \| None` | 在哪些节点后中断 |
| `debug` | `bool` | 是否启用调试模式 |
| `name` | `str \| None` | 编译后图的名称，默认 `"LangGraph"` |

---

## 6.2 编译步骤详解

### 6.2.1 步骤总览

`compile()` 方法按以下顺序执行：

1. **校验 checkpointer** — `ensure_valid_checkpointer(checkpointer)`
2. **构建 serde 白名单**（如果启用了 strict msgpack）
3. **验证图结构** — `self.validate(...)`
4. **准备输出与流 Channel**
5. **创建 CompiledStateGraph 实例**
6. **挂载节点** — `compiled.attach_node()`
7. **配置 mapper**（Pydantic/dataclass 转换器）
8. **挂载边** — `compiled.attach_edge()`
9. **挂载条件边** — `compiled.attach_branch()`
10. **最终验证** — `compiled.validate()`

我们逐一深入。

### 6.2.2 步骤 1：校验 Checkpointer

```python
checkpointer = ensure_valid_checkpointer(checkpointer)
```

这个函数检查 checkpointer 是否是合法类型：

```python
def ensure_valid_checkpointer(checkpointer: Checkpointer) -> Checkpointer:
    if checkpointer not in (None, True, False) and not isinstance(
        checkpointer, BaseCheckpointSaver
    ):
        raise TypeError(
            "Invalid checkpointer provided. Expected an instance of "
            "`BaseCheckpointSaver`, `True`, `False`, or `None`. "
            f"Received {type(checkpointer).__name__!s}. "
            "Pass a proper saver (e.g., InMemorySaver, AsyncPostgresSaver)."
        )
    return checkpointer
```

合法值包括：
- `None` — 作为子图时继承父图的 checkpointer
- `True` — 启用持久化 checkpointing
- `False` — 禁用 checkpointing
- `BaseCheckpointSaver` 实例 — 使用指定的 checkpointer

### 6.2.3 步骤 2：Serde 白名单（Strict MsgPack）

如果启用了 strict msgpack 序列化模式，编译器会遍历所有 schema 和 Channel，构建一个允许序列化的类型白名单：

```python
serde_allowlist: set[tuple[str, ...]] | None = None
if _serde.STRICT_MSGPACK_ENABLED:
    schema_types: list[type[Any]] = [
        self.state_schema,
        self.input_schema,
        self.output_schema,
    ]
    if self.context_schema is not None:
        schema_types.append(self.context_schema)
    for node in self.nodes.values():
        schema_types.append(node.input_schema)
    for branches in self.branches.values():
        for branch in branches.values():
            if branch.input_schema is not None:
                schema_types.append(branch.input_schema)
    serde_allowlist = _serde.build_serde_allowlist(
        schemas=schema_types,
        channels=self.channels,
    )
    checkpointer = _serde.apply_checkpointer_allowlist(
        checkpointer, serde_allowlist
    )
```

这保证了 checkpoint 中只会存储预期的类型，防止反序列化攻击。

---

## 6.3 图结构验证：validate()

### 6.3.1 StateGraph.validate() 方法

在创建 `CompiledStateGraph` 之前，`compile()` 调用 `self.validate()` 对声明式图做静态检查。这个方法定义在 `StateGraph` 类中：

```python
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
            raise ValueError(f"Found edge starting at unknown node '{source}'")

    if START not in all_sources:
        raise ValueError(
            "Graph must have an entrypoint: add at least one edge from START to another node"
        )
    # ...
```

### 6.3.2 验证规则一览

`validate()` 执行以下检查：

**源节点检查：**
- 收集所有边的源节点（包括静态边、条件边、Command ends）
- 确保每个源节点都存在于 `self.nodes` 中（`START` 除外）
- 确保图有入口点（`START` 必须出现在源节点中）

**目标节点检查：**
```python
# assemble targets
all_targets = {end for _, end in self._all_edges}
for start, branches in self.branches.items():
    for cond, branch in branches.items():
        if branch.ends is not None:
            for end in branch.ends.values():
                if end not in self.nodes and end != END:
                    raise ValueError(
                        f"At '{start}' node, '{cond}' branch found unknown target '{end}'"
                    )
                all_targets.add(end)
        else:
            all_targets.add(END)
            for node in self.nodes:
                if node != start:
                    all_targets.add(node)
for name, spec in self.nodes.items():
    if spec.ends:
        all_targets.update(spec.ends)
for target in all_targets:
    if target not in self.nodes and target != END:
        raise ValueError(f"Found edge ending at unknown node `{target}`")
```

关键规则：
- 静态边的终点必须是已知节点或 `END`
- 条件边的 `path_map` 中指定的目标必须是已知节点或 `END`
- 如果条件边没有 `path_map`（`ends` 为 None），则假设可以路由到任意节点
- Command 的 `ends`（destinations）中的节点也会被验证

**中断节点检查：**
```python
if interrupt:
    for node in interrupt:
        if node not in self.nodes:
            raise ValueError(f"Interrupt node `{node}` not found")
```

`interrupt_before` 和 `interrupt_after` 中指定的节点必须存在。

### 6.3.3 _all_edges 属性

`validate()` 使用 `_all_edges` 属性来获取所有边（包括等待边展开后的版本）：

```python
@property
def _all_edges(self) -> set[tuple[str, str]]:
    return self.edges | {
        (start, end) for starts, end in self.waiting_edges for start in starts
    }
```

这个属性将 `waiting_edges`（多起点边）展开为多个单独的边。例如，`([A, B], C)` 会被展开为 `{(A, C), (B, C)}`。

### 6.3.4 注意：LangGraph 不做环检测

值得注意的是，`validate()` **不检测图中的环**。这是有意为之——LangGraph 允许并鼓励循环图（例如 Agent 的思考-行动循环）。环的终止通过以下机制保证：
- `recursion_limit` 配置（默认 25 步）
- 节点通过条件边或 Command 路由到 `END`
- `IsLastStep` Managed Value 让节点知道何时该停止

---

## 6.4 准备输出与流 Channel

验证通过后，`compile()` 准备输出相关的 Channel 配置：

```python
# prepare output channels
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
stream_channels = (
    "__root__"
    if len(self.channels) == 1 and "__root__" in self.channels
    else [
        key for key, val in self.channels.items() if not is_managed_value(val)
    ]
)
```

这里有两个概念：

**output_channels** — 图的最终输出使用哪些 Channel：
- 如果 output_schema 只有一个 `__root__` Channel（即非 TypedDict 的简单类型），使用字符串 `"__root__"`
- 否则，使用 output_schema 中所有非 Managed Value 的 Channel 名称列表

**stream_channels** — 流式输出使用哪些 Channel：
- 同样的逻辑，但基于全局 `self.channels`（而非 output_schema 子集）
- 用于 `stream_mode="values"` 时输出完整状态

为什么要排除 Managed Values？因为 Managed Values（如 `IsLastStep`）是运行时计算的临时值，不应出现在图的输出或流中。

---

## 6.5 创建 CompiledStateGraph 实例

准备好所有参数后，`compile()` 创建 `CompiledStateGraph` 实例：

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

让我们拆解每个参数：

| 参数 | 值 | 说明 |
|------|------|------|
| `builder` | `self` | 保留对原始 StateGraph 的引用，用于后续操作 |
| `schema_to_mapper` | `{}` | schema 到 mapper 函数的缓存（Pydantic/dataclass 转换） |
| `nodes` | `{}` | 空字典，稍后通过 `attach_node` 填充 |
| `channels` | `{...}` | 合并 channels + managed + START Channel |
| `input_channels` | `START` | 图的输入通过 `START` Channel 进入 |
| `stream_mode` | `"updates"` | 默认流模式 |
| `output_channels` | 上一步计算的 | 图的输出 Channel |
| `stream_channels` | 上一步计算的 | 流式输出的 Channel |
| `auto_validate` | `False` | 暂时禁用自动验证，等所有节点挂载完后手动验证 |

注意 `channels` 字典中的一个特殊条目：

```python
START: EphemeralValue(self.input_schema)
```

`START` 作为一个 `EphemeralValue` Channel 存在——它是临时的（ephemeral），在一个 super step 结束后就被清空。用户的输入被写入这个 Channel，触发 START 节点执行。

### 6.5.1 CompiledStateGraph 类结构

`CompiledStateGraph` 继承自 `Pregel`：

```python
class CompiledStateGraph(
    Pregel[StateT, ContextT, InputT, OutputT],
    Generic[StateT, ContextT, InputT, OutputT],
):
    builder: StateGraph[StateT, ContextT, InputT, OutputT]
    schema_to_mapper: dict[type[Any], Callable[[Any], Any] | None]
    _output_mapper: Callable[[Any], Any] | None
    _state_mapper: Callable[[Any], Any] | None
```

它在 `Pregel` 的基础上增加了：
- `builder` — 对原始 StateGraph 的引用
- `schema_to_mapper` — schema 到类型转换函数的缓存
- `_output_mapper` / `_state_mapper` — 输出/状态的类型转换函数

### 6.5.2 CompiledStateGraph vs CompiledGraph

在 LangGraph 源码中，你可能会看到两个类名：

- `CompiledStateGraph` — StateGraph 编译后的产物，继承自 `Pregel`
- `Pregel` — 基础执行引擎，实现了 `invoke`/`stream` 等方法

`CompiledStateGraph` 并不添加新的执行逻辑，它主要负责：
1. 将 StateGraph 的声明式结构转换为 `PregelNode`
2. 处理 schema 映射（TypedDict -> Pydantic model 等）
3. 提供 JSON Schema 生成方法

---

## 6.6 挂载节点：attach_node

### 6.6.1 attach_node 概览

`attach_node` 是编译过程中最关键的方法之一。它将 `StateNodeSpec`（声明式节点规格）转换为 `PregelNode`（可执行节点容器）。`compile()` 对每个节点依次调用：

```python
compiled.attach_node(START, None)
for key, node in self.nodes.items():
    compiled.attach_node(key, node)
```

### 6.6.2 START 节点的特殊处理

当 `key == START` 时，`attach_node` 创建一个特殊的 START PregelNode：

```python
if key == START:
    output_keys = [
        k
        for k, v in self.builder.schemas[self.builder.input_schema].items()
        if not is_managed_value(v)
    ]
    # ...
    self.nodes[key] = PregelNode(
        tags=[TAG_HIDDEN],
        triggers=[START],
        channels=START,
        writers=[ChannelWrite(write_entries)],
    )
```

START 节点的特点：
- `tags=[TAG_HIDDEN]` — 在流式输出中隐藏
- `triggers=[START]` — 当 `START` Channel 被写入时触发
- `channels=START` — 从 `START` Channel 读取输入
- `writers` — 将输入分发到各个状态 Channel

### 6.6.3 普通节点的 PregelNode 构建

对于普通节点，`attach_node` 的逻辑更复杂。首先确定输入 Channel：

```python
input_schema = node.input_schema if node else self.builder.state_schema
input_channels = list(self.builder.schemas[input_schema])
is_single_input = len(input_channels) == 1 and "__root__" in input_channels
```

然后确定 mapper（类型转换函数）：

```python
if input_schema in self.schema_to_mapper:
    mapper = self.schema_to_mapper[input_schema]
else:
    mapper = _pick_mapper(input_channels, input_schema)
    self.schema_to_mapper[input_schema] = mapper
```

`_pick_mapper` 函数决定是否需要将字典状态转换为 Pydantic model 或 dataclass：

```python
def _pick_mapper(
    state_keys: Sequence[str], schema: type[Any]
) -> Callable[[Any], Any] | None:
    if state_keys == ["__root__"]:
        return None
    if isclass(schema) and (issubclass(schema, BaseModel) or is_dataclass(schema)):
        return partial(_coerce_state, schema)
    return None
```

如果 schema 是 Pydantic model 或 dataclass，mapper 会将字典转换为对应的实例：

```python
def _coerce_state(schema: type[_S], input: dict[str, Any]) -> _S:
    return schema(**input)
```

接下来，创建触发 Channel 和 PregelNode：

```python
branch_channel = _CHANNEL_BRANCH_TO.format(key)  # "branch:to:{node_name}"
self.channels[branch_channel] = (
    LastValueAfterFinish(Any)
    if node.defer
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

### 6.6.4 branch:to:{node} 触发 Channel

这是 LangGraph 编译过程中最重要的设计决策之一。每个节点都有一个专属的触发 Channel，命名格式为 `branch:to:{node_name}`。

```python
_CHANNEL_BRANCH_TO = "branch:to:{}"
```

这个 Channel 的类型取决于节点是否为 defer 节点：
- **普通节点**：`EphemeralValue(Any, guard=False)` — 临时值，每个 super step 后清空
- **defer 节点**：`LastValueAfterFinish(Any)` — 保留值直到图执行完成

当边（无论是静态边还是条件边）想要触发某个节点时，它只需向对应的 `branch:to:{node}` Channel 写入一个值。这种统一机制使得所有边类型最终都汇聚到同一种触发方式。

### 6.6.5 write_entries：状态更新与路由控制

每个节点的 `writers` 中都包含一个 `ChannelWrite`，它负责处理节点的返回值。这个 `ChannelWrite` 包含两个 `ChannelWriteTupleEntry`：

```python
write_entries: tuple[ChannelWriteEntry | ChannelWriteTupleEntry, ...] = (
    ChannelWriteTupleEntry(
        mapper=_get_root if output_keys == ["__root__"] else _get_updates
    ),
    ChannelWriteTupleEntry(
        mapper=_control_branch,
        static=_control_static(node.ends)
        if node is not None and node.ends is not None
        else None,
    ),
)
```

**第一个 entry** — 状态更新提取器：
- 对于 `__root__` Channel：使用 `_get_root` 函数
- 对于字典状态：使用 `_get_updates` 函数

**第二个 entry** — 路由控制提取器：
- 使用 `_control_branch` 函数从 Command/Send 中提取路由信息
- `static` 字段用于图渲染，声明可能的路由目标

### 6.6.6 _get_updates 函数

这个函数是节点输出到状态更新的核心转换逻辑，定义为 `attach_node` 内的闭包：

```python
def _get_updates(
    input: None | dict | Any,
) -> Sequence[tuple[str, Any]] | None:
    if input is None:
        return None
    elif isinstance(input, dict):
        return [(k, v) for k, v in input.items() if k in output_keys]
    elif isinstance(input, Command):
        if input.graph == Command.PARENT:
            return None
        return [
            (k, v) for k, v in input._update_as_tuples() if k in output_keys
        ]
    elif (
        isinstance(input, (list, tuple))
        and input
        and any(isinstance(i, Command) for i in input)
    ):
        updates: list[tuple[str, Any]] = []
        for i in input:
            if isinstance(i, Command):
                if i.graph == Command.PARENT:
                    continue
                updates.extend(
                    (k, v) for k, v in i._update_as_tuples() if k in output_keys
                )
            else:
                updates.extend(_get_updates(i) or ())
        return updates
    elif (t := type(input)) and get_cached_annotated_keys(t):
        return get_update_as_tuples(input, output_keys)
    else:
        raise InvalidUpdateError(f"Expected dict, got {input}")
```

它支持多种节点返回值格式：
1. `None` — 不更新状态
2. `dict` — 过滤出合法的 key 后作为更新
3. `Command` — 提取 `update` 字段
4. `list[Command]` — 合并多个 Command 的更新
5. Pydantic model / dataclass — 通过 `get_update_as_tuples` 转换

注意 `output_keys` 的过滤——只有声明在 schema 中的 key 才会被写入 Channel，未知的 key 会被静默忽略。

---

## 6.7 PregelNode 的构建

### 6.7.1 PregelNode 类定义

`PregelNode` 是编译后节点的容器，定义在 `/tmp/langgraph-src/libs/langgraph/langgraph/pregel/_read.py`：

```python
class PregelNode:
    """A node in a Pregel graph. This won't be invoked as a runnable by the graph
    itself, but instead acts as a container for the components necessary to make
    a PregelExecutableTask for a node."""

    channels: str | list[str]
    triggers: list[str]
    mapper: Callable[[Any], Any] | None
    writers: list[Runnable]
    bound: Runnable[Any, Any]
    retry_policy: Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    tags: Sequence[str] | None
    metadata: Mapping[str, Any] | None
    subgraphs: Sequence[PregelProtocol]
```

### 6.7.2 channels vs triggers

这两个字段是 PregelNode 最核心的概念，理解它们对于理解 Pregel 执行模型至关重要：

**triggers（触发器）：**
```python
triggers: list[str]
```

一个 Channel 名称列表。当**任何一个**触发 Channel 在某个 super step 中被写入时，该节点就会在下一个 super step 中被调度执行。

对于普通 StateGraph 节点，triggers 通常是 `["branch:to:{node_name}"]`。但节点也可以有多个触发器（例如等待边会额外添加 `NamedBarrierValue` Channel 作为触发器）。

**channels（输入 Channel）：**
```python
channels: str | list[str]
```

决定节点执行时从哪些 Channel 读取输入数据：
- 如果是字符串（如 `"__root__"`），节点的输入就是该 Channel 的值
- 如果是列表（如 `["messages", "context", "result"]`），节点的输入是一个字典，key 为 Channel 名，value 为 Channel 值

### 6.7.3 mapper：输入转换

```python
mapper: Callable[[Any], Any] | None
```

在 Channel 数据读取后、传入 `bound` 之前执行的转换函数。主要用于将字典状态转换为 Pydantic model 或 dataclass 实例。

如果 mapper 为 `None`，则直接传递原始字典。

### 6.7.4 bound：核心执行逻辑

```python
bound: Runnable[Any, Any]
```

节点的实际执行逻辑——就是用户传入的函数（已被 `coerce_to_runnable` 包装为 Runnable）。

如果用户没有指定 bound（例如 START 节点），则使用 `DEFAULT_BOUND`：

```python
DEFAULT_BOUND = RunnableCallable(lambda input: input)
```

即一个恒等函数（identity function）。

### 6.7.5 writers：输出处理管道

```python
writers: list[Runnable]
```

一个 Runnable 列表，在 `bound` 执行完毕后依次执行。每个 writer 负责将节点的输出写入相应的 Channel。

对于 StateGraph 节点，writers 通常包含：
1. 一个 `ChannelWrite` — 处理状态更新和路由控制
2. 可能有额外的 `ChannelWrite` — 由静态边或条件边添加

### 6.7.6 node 属性：组装完整的执行管道

PregelNode 有一个 `node` cached property，它将 `bound` 和 `writers` 组装为一个完整的执行管道：

```python
@cached_property
def node(self) -> Runnable[Any, Any] | None:
    """Get a runnable that combines `bound` and `writers`."""
    writers = self.flat_writers
    if self.bound is DEFAULT_BOUND and not writers:
        return None
    elif self.bound is DEFAULT_BOUND and len(writers) == 1:
        return writers[0]
    elif self.bound is DEFAULT_BOUND:
        return RunnableSeq(*writers)
    elif writers:
        return RunnableSeq(self.bound, *writers)
    else:
        return self.bound
```

`RunnableSeq` 是一个顺序执行的 Runnable 链。最终的执行管道是：

```
输入 -> bound（用户逻辑）-> writer1 -> writer2 -> ... -> writerN
```

### 6.7.7 flat_writers：写入优化

`flat_writers` 属性对连续的 `ChannelWrite` 做了合并优化：

```python
@cached_property
def flat_writers(self) -> list[Runnable]:
    """Get writers with optimizations applied. Dedupes consecutive ChannelWrites."""
    writers = self.writers.copy()
    while (
        len(writers) > 1
        and isinstance(writers[-1], ChannelWrite)
        and isinstance(writers[-2], ChannelWrite)
    ):
        writers[-2] = ChannelWrite(
            writes=writers[-2].writes + writers[-1].writes,
        )
        writers.pop()
    return writers
```

如果两个连续的 writer 都是 `ChannelWrite`，它们会被合并为一个，减少执行开销。

### 6.7.8 subgraphs：子图检测

PregelNode 在构造时会自动检测 `bound` 中是否包含子图：

```python
if subgraphs is not None:
    self.subgraphs = subgraphs
elif self.bound is not DEFAULT_BOUND:
    try:
        subgraph = find_subgraph_pregel(self.bound)
    except Exception:
        subgraph = None
    if subgraph:
        self.subgraphs = [subgraph]
    else:
        self.subgraphs = []
```

这用于支持子图的 streaming 和状态快照功能。

---

## 6.8 挂载边：attach_edge

### 6.8.1 静态边的挂载

`compile()` 遍历所有静态边，对每条边调用 `attach_edge`：

```python
for start, end in self.edges:
    compiled.attach_edge(start, end)

for starts, end in self.waiting_edges:
    compiled.attach_edge(starts, end)
```

`attach_edge` 方法的实现：

```python
def attach_edge(self, starts: str | Sequence[str], end: str) -> None:
    if isinstance(starts, str):
        # subscribe to start channel
        if end != END:
            self.nodes[starts].writers.append(
                ChannelWrite(
                    (ChannelWriteEntry(_CHANNEL_BRANCH_TO.format(end), None),)
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

### 6.8.2 单起点边的处理

对于简单的 `A -> B` 边：

1. 在 A 的 `writers` 中添加一个 `ChannelWrite`
2. 这个 `ChannelWrite` 向 `branch:to:B` Channel 写入 `None`
3. 当 A 执行完毕时，`branch:to:B` Channel 被更新
4. B 的 `triggers` 包含 `branch:to:B`，因此 B 会在下一个 super step 被触发

如果 `end == END`，则不做任何操作——因为 `END` 不是一个真正的节点，没有对应的触发 Channel。图在没有新节点被触发时自动结束。

### 6.8.3 等待边的处理

等待边（多起点边）的处理更复杂：

```python
channel_name = f"join:{'+'.join(starts)}:{end}"
```

例如 `add_edge(["A", "B"], "C")` 会创建一个名为 `join:A+B:C` 的 Channel。

这个 Channel 的类型是 `NamedBarrierValue`：

```python
self.channels[channel_name] = NamedBarrierValue(str, set(starts))
```

`NamedBarrierValue` 是一种特殊的 Channel，它记录了一组"需要签到"的名字。只有当所有名字都签到后，Channel 才算更新完毕（被标记为可用）。

每个起点节点在完成时写入自己的名字：

```python
for start in starts:
    self.nodes[start].writers.append(
        ChannelWrite((ChannelWriteEntry(channel_name, start),))
    )
```

终点节点将这个 barrier Channel 加入触发器：

```python
self.nodes[end].triggers.append(channel_name)
```

工作流程：
1. A 执行完毕 -> 写入 `"A"` 到 `join:A+B:C`
2. B 执行完毕 -> 写入 `"B"` 到 `join:A+B:C`
3. `NamedBarrierValue` 检测到 `{"A", "B"}` 已全部签到
4. Channel 更新版本号 -> C 的触发器被激活
5. C 在下一个 super step 执行

---

## 6.9 挂载条件边：attach_branch

### 6.9.1 方法定义

```python
def attach_branch(
    self, start: str, name: str, branch: BranchSpec, *, with_reader: bool = True
) -> None:
```

### 6.9.2 Reader 的创建

条件边需要一个 reader 来在路由时读取最新状态：

```python
if with_reader:
    schema = branch.input_schema or (
        self.builder.nodes[start].input_schema
        if start in self.builder.nodes
        else self.builder.state_schema
    )
    channels = list(self.builder.schemas[schema])
    if schema in self.schema_to_mapper:
        mapper = self.schema_to_mapper[schema]
    else:
        mapper = _pick_mapper(channels, schema)
        self.schema_to_mapper[schema] = mapper
    reader: Callable[[RunnableConfig], Any] | None = partial(
        ChannelRead.do_read,
        select=channels[0] if channels == ["__root__"] else channels,
        fresh=True,
        mapper=mapper,
    )
else:
    reader = None
```

Reader 使用 `ChannelRead.do_read` 读取最新的 Channel 值。关键参数：
- `select` — 要读取的 Channel 名称
- `fresh=True` — 读取最新值（在同一 super step 内可能已被其他节点更新）
- `mapper` — 类型转换函数

### 6.9.3 Writer 函数

`attach_branch` 内部定义了一个 `get_writes` 函数，用于将路由目标转换为 Channel 写入：

```python
def get_writes(
    packets: Sequence[str | Send], static: bool = False
) -> Sequence[ChannelWriteEntry | Send]:
    writes = [
        (
            ChannelWriteEntry(
                p if p == END else _CHANNEL_BRANCH_TO.format(p), None
            )
            if not isinstance(p, Send)
            else p
        )
        for p in packets
        if (True if static else p != END)
    ]
    if not writes:
        return []
    return writes
```

对于字符串目标（节点名），创建 `ChannelWriteEntry` 写入 `branch:to:{node}`。对于 `Send` 对象，直接保留。

### 6.9.4 挂载到源节点

最后，条件边被包装为一个 Runnable 并添加到源节点的 writers 中：

```python
self.nodes[start].writers.append(branch.run(get_writes, reader))
```

`BranchSpec.run` 方法（见第 5 章）返回一个 `RunnableCallable`，它在运行时执行路由函数、调用 `_finish` 处理结果、并通过 `get_writes` 转换为 Channel 写入。

---

## 6.10 ChannelRead 与 ChannelWrite

### 6.10.1 ChannelRead

`ChannelRead` 是一个可运行对象，用于从 Channel 中读取数据。它定义在 `/tmp/langgraph-src/libs/langgraph/langgraph/pregel/_read.py`：

```python
class ChannelRead(RunnableCallable):
    """Implements the logic for reading state from CONFIG_KEY_READ.
    Usable both as a runnable as well as a static method to call imperatively."""

    channel: str | list[str]
    fresh: bool = False
    mapper: Callable[[Any], Any] | None = None
```

其核心逻辑在 `do_read` 静态方法中：

```python
@staticmethod
def do_read(
    config: RunnableConfig,
    *,
    select: str | list[str],
    fresh: bool = False,
    mapper: Callable[[Any], Any] | None = None,
) -> Any:
    try:
        read: READ_TYPE = config[CONF][CONFIG_KEY_READ]
    except KeyError:
        raise RuntimeError(
            "Not configured with a read function"
            "Make sure to call in the context of a Pregel process"
        )
    if mapper:
        return mapper(read(select, fresh))
    else:
        return read(select, fresh)
```

`CONFIG_KEY_READ` 是一个由 Pregel 引擎在运行时注入到 config 中的函数。这个函数知道如何从当前 super step 的 Channel 快照中读取数据。

### 6.10.2 ChannelWrite

`ChannelWrite` 是 `ChannelRead` 的对偶，负责将数据写入 Channel。它的核心方法 `do_write` 从 config 中获取 `CONFIG_KEY_SEND` 函数：

```python
@staticmethod
def do_write(
    config: RunnableConfig,
    writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
) -> None:
    # ...
    write: TYPE_SEND = config[CONF][CONFIG_KEY_SEND]
    write(writes_to_send)
```

### 6.10.3 ChannelWriteEntry vs ChannelWriteTupleEntry

LangGraph 有两种写入条目：

**ChannelWriteEntry** — 简单的单 Channel 写入：
```python
class ChannelWriteEntry(NamedTuple):
    channel: str        # 目标 Channel
    value: Any = PASSTHROUGH  # 写入的值（PASSTHROUGH 表示使用节点输出）
    skip_none: bool = False   # 是否跳过 None 值
    mapper: Callable | None = None  # 值转换函数
```

**ChannelWriteTupleEntry** — 动态的多 Channel 写入：
```python
class ChannelWriteTupleEntry(NamedTuple):
    mapper: Callable[[Any], Sequence[tuple[str, Any]] | None]
    value: Any = PASSTHROUGH
    static: Sequence[tuple[str, Any, str | None]] | None = None
```

`ChannelWriteTupleEntry` 的 `mapper` 函数接受节点输出，返回 `(channel_name, value)` 元组的序列。这种设计允许一个写入条目动态地向多个 Channel 写入——正是状态更新（`_get_updates`）和路由控制（`_control_branch`）所需要的。

---

## 6.11 最终验证：validate_graph

### 6.11.1 compile() 的最后一步

```python
return compiled.validate()
```

`compiled.validate()` 最终调用 `validate_graph` 函数（定义在 `/tmp/langgraph-src/libs/langgraph/langgraph/pregel/_validate.py`），对编译后的 Pregel 图做低级别验证：

```python
def validate_graph(
    nodes: Mapping[str, PregelNode],
    channels: dict[str, BaseChannel],
    managed: ManagedValueMapping,
    input_channels: str | Sequence[str],
    output_channels: str | Sequence[str],
    stream_channels: str | Sequence[str] | None,
    interrupt_after_nodes: All | Sequence[str],
    interrupt_before_nodes: All | Sequence[str],
) -> None:
```

### 6.11.2 验证规则

**保留名检查：**
```python
for chan in channels:
    if chan in RESERVED:
        raise ValueError(f"Channel name '{chan}' is reserved")
for name in managed:
    if name in RESERVED:
        raise ValueError(f"Managed name '{name}' is reserved")
for name, node in nodes.items():
    if name in RESERVED:
        raise ValueError(f"Node name '{name}' is reserved")
```

`RESERVED` 包含 LangGraph 内部使用的名称，用户不能使用。

**Channel 存在性检查：**
```python
if isinstance(node, PregelNode):
    subscribed_channels.update(node.triggers)
    if isinstance(node.channels, str):
        if node.channels not in channels:
            raise ValueError(
                f"Node {name} reads channel '{node.channels}' "
                f"not in known channels: ..."
            )
    else:
        for chan in node.channels:
            if chan not in channels and chan not in managed:
                raise ValueError(...)
```

确保每个节点引用的 Channel 都存在于 `channels` 或 `managed` 字典中。

**触发 Channel 检查：**
```python
for chan in subscribed_channels:
    if chan not in channels:
        raise ValueError(
            f"Subscribed channel '{chan}' not in known channels: ..."
        )
```

确保所有触发 Channel 都存在。

**输入 Channel 检查：**
```python
if isinstance(input_channels, str):
    if input_channels not in channels:
        raise ValueError(f"Input channel '{input_channels}' not in known channels: ...")
    if input_channels not in subscribed_channels:
        raise ValueError(
            f"Input channel {input_channels} is not subscribed to by any node"
        )
```

输入 Channel 必须存在，且至少有一个节点订阅了它。对于 StateGraph，输入 Channel 是 `START`，而 START 节点订阅了它。

**输出 Channel 检查：**
```python
for chan in all_output_channels:
    if chan not in channels:
        raise ValueError(f"Output channel '{chan}' not in known channels: ...")
```

所有输出和流 Channel 都必须存在。

**中断节点检查：**
```python
if interrupt_after_nodes != "*":
    for n in interrupt_after_nodes:
        if n not in nodes:
            raise ValueError(f"Node {n} not in nodes")
if interrupt_before_nodes != "*":
    for n in interrupt_before_nodes:
        if n not in nodes:
            raise ValueError(f"Node {n} not in nodes")
```

### 6.11.3 validate_keys 辅助函数

```python
def validate_keys(
    keys: str | Sequence[str] | None,
    channels: Mapping[str, Any],
) -> None:
    if isinstance(keys, str):
        if keys not in channels:
            raise ValueError(f"Key {keys} not in channels")
    elif keys is not None:
        for chan in keys:
            if chan not in channels:
                raise ValueError(f"Key {chan} not in channels")
```

这个函数用于验证 key 是否在 channels 中存在。

---

## 6.12 Mapper 配置

### 6.12.1 Output Mapper 与 State Mapper

在节点挂载完成后，`compile()` 配置输出和状态的 mapper：

```python
compiled._output_mapper = _pick_mapper(
    list(output_channels)
    if isinstance(output_channels, list)
    else [output_channels],
    self.output_schema,
)
compiled._state_mapper = _pick_mapper(
    list(stream_channels)
    if isinstance(stream_channels, list)
    else [stream_channels],
    self.state_schema,
)
```

这些 mapper 用于在 v2 stream API 中将字典输出转换为用户的 schema 类型（Pydantic model 或 dataclass）。

### 6.12.2 schema_to_mapper 缓存

`schema_to_mapper` 字典缓存了每种 schema 对应的 mapper 函数，避免重复计算：

```python
if input_schema in self.schema_to_mapper:
    mapper = self.schema_to_mapper[input_schema]
else:
    mapper = _pick_mapper(input_channels, input_schema)
    self.schema_to_mapper[input_schema] = mapper
```

如果多个节点使用相同的 `input_schema`，它们会共享同一个 mapper 实例。

---

## 6.13 Checkpoint 迁移

### 6.13.1 _migrate_checkpoint 方法

`CompiledStateGraph` 实现了 `_migrate_checkpoint` 方法，用于将旧版本的 checkpoint 迁移到新的 Channel 布局：

```python
def _migrate_checkpoint(self, checkpoint: Checkpoint) -> None:
    super()._migrate_checkpoint(checkpoint)

    values = checkpoint["channel_values"]
    versions = checkpoint["channel_versions"]
    seen = checkpoint["versions_seen"]

    if not versions:
        return

    if checkpoint["v"] >= 3:
        return
    # ...
```

### 6.13.2 迁移历史

LangGraph 的 Channel 命名经历了几次变化：

1. **早期**：节点触发 Channel 直接以节点名命名（如 `"node_a"`）
2. **中期**：使用 `start:{node}` 格式（如 `"start:node_a"`）
3. **后期**：使用 `branch:{source}:{condition}:{target}` 格式
4. **当前**：统一为 `branch:to:{node}` 格式

`_migrate_checkpoint` 处理了从旧格式到新格式的迁移，包括：
- `start:{node}` -> `branch:to:{node}`
- `branch:{source}:{condition}:{node}` -> `branch:to:{node}`
- `{node}` -> `branch:to:{node}`

迁移时需要更新三个数据结构：
- `channel_values` — Channel 的实际值
- `channel_versions` — Channel 的版本号
- `versions_seen` — 每个节点已看到的 Channel 版本

---

## 6.14 编译过程完整示例

让我们用一个具体的例子跟踪完整的编译过程：

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator

class State(TypedDict):
    query: str
    result: str
    scores: Annotated[list[float], operator.add]

def analyzer(state: State) -> dict:
    return {"scores": [0.9]}

def router(state: State) -> Literal["fast", "slow"]:
    if state["scores"][-1] > 0.5:
        return "fast"
    return "slow"

def fast_path(state: State) -> dict:
    return {"result": "fast result"}

def slow_path(state: State) -> dict:
    return {"result": "slow result"}

builder = StateGraph(State)
builder.add_node(analyzer)
builder.add_node(fast_path)
builder.add_node(slow_path)
builder.add_edge(START, "analyzer")
builder.add_conditional_edges("analyzer", router)
builder.add_edge("fast_path", END)
builder.add_edge("slow_path", END)
graph = builder.compile()
```

编译后产生的数据结构：

**channels 字典：**
```python
{
    "query": LastValue(str),             # State 字段
    "result": LastValue(str),            # State 字段
    "scores": BinaryOperatorAggregate(list, operator.add),  # State 字段
    START: EphemeralValue(State),        # 输入 Channel
    "branch:to:analyzer": EphemeralValue(Any),   # 触发 Channel
    "branch:to:fast_path": EphemeralValue(Any),  # 触发 Channel
    "branch:to:slow_path": EphemeralValue(Any),  # 触发 Channel
}
```

**nodes 字典：**
```python
{
    START: PregelNode(
        triggers=[START],
        channels=START,
        writers=[ChannelWrite(...)],  # 将输入分发到 state channels
        tags=[TAG_HIDDEN],
    ),
    "analyzer": PregelNode(
        triggers=["branch:to:analyzer"],
        channels=["query", "result", "scores"],
        bound=<RunnableCallable: analyzer>,
        writers=[
            ChannelWrite(write_entries),   # 状态更新 + 路由控制
            ChannelWrite(...)              # 来自 attach_branch (router)
        ],
    ),
    "fast_path": PregelNode(
        triggers=["branch:to:fast_path"],
        channels=["query", "result", "scores"],
        bound=<RunnableCallable: fast_path>,
        writers=[ChannelWrite(write_entries)],  # 状态更新（无额外 edge writer，因 end=END）
    ),
    "slow_path": PregelNode(
        triggers=["branch:to:slow_path"],
        channels=["query", "result", "scores"],
        bound=<RunnableCallable: slow_path>,
        writers=[ChannelWrite(write_entries)],
    ),
}
```

**执行流程：**

```
Step 0: 用户输入 -> 写入 START Channel
Step 1: START node 触发
  - 读取 START Channel
  - 写入 query/result/scores Channels
  - 写入 branch:to:analyzer Channel (由 add_edge(START, "analyzer") 添加)
Step 2: analyzer 节点触发
  - 读取 query/result/scores Channels
  - 执行 analyzer 函数 -> {"scores": [0.9]}
  - writer 1: 将 scores=[0.9] 写入 scores Channel
  - writer 2 (条件边): 执行 router -> "fast"
    -> 写入 branch:to:fast_path Channel
Step 3: fast_path 节点触发
  - 读取 query/result/scores Channels
  - 执行 fast_path 函数 -> {"result": "fast result"}
  - writer: 将 result="fast result" 写入 result Channel
  - end=END，无额外触发
Step 4: 无新节点被触发，图结束
  - 输出 output_channels 的内容
```

---

## 6.15 CompiledStateGraph 的额外能力

### 6.15.1 JSON Schema 生成

`CompiledStateGraph` 提供了输入/输出的 JSON Schema 生成方法：

```python
def get_input_jsonschema(
    self, config: RunnableConfig | None = None
) -> dict[str, Any]:
    return _get_json_schema(
        typ=self.builder.input_schema,
        schemas=self.builder.schemas,
        channels=self.builder.channels,
        name=self.get_name("Input"),
    )

def get_output_jsonschema(
    self, config: RunnableConfig | None = None
) -> dict[str, Any]:
    return _get_json_schema(
        typ=self.builder.output_schema,
        schemas=self.builder.schemas,
        channels=self.builder.channels,
        name=self.get_name("Output"),
    )
```

`_get_json_schema` 函数根据 schema 类型选择不同的生成策略：
- Pydantic model -> `model_json_schema()`
- TypedDict -> `TypeAdapter(typ).json_schema()`
- 简单类型 -> 创建临时 Pydantic model

### 6.15.2 作为 Runnable 使用

由于 `CompiledStateGraph` 继承自 `Pregel`，而 `Pregel` 实现了 `Runnable` 接口，因此编译后的图可以直接作为 Runnable 使用：

```python
# invoke
result = graph.invoke({"query": "hello"})

# stream
for event in graph.stream({"query": "hello"}):
    print(event)

# async
result = await graph.ainvoke({"query": "hello"})

# batch
results = graph.batch([{"query": "a"}, {"query": "b"}])
```

这也意味着编译后的图可以作为另一个图的节点：

```python
parent = StateGraph(ParentState)
parent.add_node("subgraph", graph)  # 直接使用编译后的图作为节点
```

---

## 6.16 defer 节点的特殊编译

### 6.16.1 什么是 defer 节点

`defer=True` 的节点会在图即将结束时（最后一个 super step）才执行。这在需要"清理"或"总结"操作时很有用。

### 6.16.2 defer 的编译差异

defer 节点在编译时有两个关键差异：

**触发 Channel 类型不同：**
```python
self.channels[branch_channel] = (
    LastValueAfterFinish(Any)  # defer=True
    if node.defer
    else EphemeralValue(Any, guard=False)  # defer=False (普通)
)
```

- `EphemeralValue` — 每个 super step 后清空，立即触发
- `LastValueAfterFinish` — 保持值直到图结束，在最后阶段触发

**等待边 Channel 类型不同：**
```python
if self.builder.nodes[end].defer:
    self.channels[channel_name] = NamedBarrierValueAfterFinish(
        str, set(starts)
    )
else:
    self.channels[channel_name] = NamedBarrierValue(str, set(starts))
```

`NamedBarrierValueAfterFinish` 与 `NamedBarrierValue` 的区别在于前者在图结束阶段才检查 barrier 条件。

---

## 6.17 编译过程的设计哲学

回顾整个编译过程，我们可以总结几个核心设计原则：

### 6.17.1 声明与执行分离

`StateGraph`（Builder）和 `CompiledStateGraph`（执行器）完全分离。Builder 负责收集用户声明，不关心执行细节；编译器负责翻译，不暴露给用户。

### 6.17.2 一切皆 Channel 写入

编译过程将所有类型的边（静态边、条件边、等待边）统一转换为 Channel 写入操作。这使得 Pregel 执行引擎只需要一种触发机制——监听 Channel 更新。

### 6.17.3 组合优于继承

PregelNode 通过组合 `bound`（用户逻辑）和 `writers`（输出处理）来构建完整的执行管道，而不是通过继承。这使得每个组件可以独立测试和替换。

### 6.17.4 延迟验证

编译过程分两阶段验证：
1. `StateGraph.validate()` — 在创建 CompiledStateGraph 前检查声明式结构
2. `validate_graph()` — 在所有节点挂载后检查编译后的执行结构

这两阶段覆盖了不同层面的问题。

---

## 本章要点

1. **compile() 是桥梁**：`StateGraph.compile()` 将声明式的 Builder 结构转换为可执行的 `CompiledStateGraph`（继承自 `Pregel`）。编译过程包括验证、Channel 准备、节点挂载、边挂载、最终验证等步骤。

2. **两阶段验证**：`StateGraph.validate()` 检查声明级别的完整性（入口存在、节点存在、目标合法），`validate_graph()` 检查编译级别的一致性（Channel 存在、订阅正确、输入/输出合法）。LangGraph 有意不做环检测——循环图是核心特性。

3. **PregelNode 是编译核心产物**：每个声明式节点被转换为一个 `PregelNode`，包含 `triggers`（触发 Channel）、`channels`（输入 Channel）、`bound`（用户逻辑）、`writers`（输出处理管道）。`triggers` 和 `channels` 是两个不同的概念——前者决定节点何时被触发，后者决定节点读取哪些数据。

4. **branch:to:{node} 统一触发机制**：每个节点有一个专属的触发 Channel `branch:to:{node_name}`。无论是静态边、条件边还是 Command 路由，最终都通过写入这个 Channel 来触发目标节点。这种统一机制大大简化了执行引擎的调度逻辑。

5. **ChannelWrite 与 ChannelWriteTupleEntry 实现动态写入**：`ChannelWriteTupleEntry` 的 `mapper` 函数使得单个写入操作可以动态地向多个 Channel 写入数据——`_get_updates` 负责状态更新，`_control_branch` 负责路由控制。

6. **CompiledStateGraph 继承自 Pregel**：编译后的图直接具备 `invoke`、`stream`、`ainvoke` 等 `Runnable` 接口方法，也可以作为另一个图的子图节点使用。它额外提供了 schema mapper（Pydantic/dataclass 转换）和 JSON Schema 生成能力。

7. **等待边使用 NamedBarrierValue**：多起点边（`add_edge([A, B], C)`）编译为 `NamedBarrierValue` Channel，只有当所有起点节点都完成写入后，终点节点才会被触发。defer 节点使用 `AfterFinish` 变体，延迟到图结束阶段才触发。
