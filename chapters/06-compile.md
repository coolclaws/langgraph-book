# 第 6 章 编译：从声明图到可执行 Pregel

前几章中，我们逐一剖析了 State、channel、节点与边的声明式 API。但 `StateGraph` 本身并不能运行——它只是一棵**声明树**。调用 `compile()` 之后，声明树才被转化为可执行的 `CompiledStateGraph`（继承自 `Pregel`）。本章将完整追踪这一编译过程，揭示从用户的 `builder.compile()` 到最终 `Pregel` 实例之间发生的每一步。

## 6.1 compile() 方法概览

`StateGraph.compile()` 是整个编译流程的入口。它的签名如下：

```python
# langgraph/graph/state.py

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

其中 `checkpointer` 支持 `None`（继承父图）、`False`（禁用）或 `BaseCheckpointSaver` 实例；`store` 提供长期存储；`interrupt_before/after` 用于 human-in-the-loop 中断。

编译的完整流程可分为以下阶段：

1. 序列化白名单构建（strict msgpack 模式）
2. 图结构验证（`validate`）
3. 输出 channel 计算
4. 创建 `CompiledStateGraph` 实例
5. 挂载节点（`attach_node`）
6. 挂载边（`attach_edge`）
7. 挂载条件分支（`attach_branch`）
8. 最终验证（`validate`）

## 6.2 第一步：序列化白名单

当启用 strict msgpack 序列化时，框架收集所有 schema 类型（state_schema、input_schema、output_schema、context_schema、各节点的 input_schema、各分支的 input_schema），调用 `_serde.build_serde_allowlist` 构建白名单。这确保 checkpoint 序列化时只允许已知类型，防止任意对象反序列化带来的安全隐患。

## 6.3 第二步：图结构验证——validate()

`compile()` 调用 `self.validate()` 进行声明式层面的验证。这是编译前的"静态检查"。

```python
# langgraph/graph/state.py

def validate(self, interrupt: Sequence[str] | None = None) -> Self:
    # 收集所有边的起始节点
    all_sources = {src for src, _ in self._all_edges}
    for start, branches in self.branches.items():
        all_sources.add(start)
    for name, spec in self.nodes.items():
        if spec.ends:
            all_sources.add(name)

    # 验证起始节点存在
    for source in all_sources:
        if source not in self.nodes and source != START:
            raise ValueError(f"Found edge starting at unknown node '{source}'")

    # 必须有入口点
    if START not in all_sources:
        raise ValueError(
            "Graph must have an entrypoint: add at least one edge from START to another node"
        )

    # 收集并验证所有目标节点
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

    for target in all_targets:
        if target not in self.nodes and target != END:
            raise ValueError(f"Found edge ending at unknown node `{target}`")

    # 验证 interrupt 节点存在
    if interrupt:
        for node in interrupt:
            if node not in self.nodes:
                raise ValueError(f"Interrupt node `{node}` not found")

    self.compiled = True
    return self
```

这里的验证集中在**拓扑合法性**：

- 所有边的源节点和目标节点必须已注册
- 图必须有从 `START` 出发的入口
- 条件分支的目标节点必须存在
- interrupt 节点必须存在

注意，此处**没有**做循环检测。LangGraph 允许循环（这是它作为 Agent 框架的核心优势），循环控制通过 `recursion_limit` 在运行时实现。

## 6.4 第三步：输出 channel 计算

验证通过后，框架计算 `output_channels` 和 `stream_channels`：

```python
# langgraph/graph/state.py - compile()

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

当 State 只有 `__root__` 一个 key 时进入单根模式；否则将 schema 中所有非 `ManagedValue` 的 key 作为 output channels。`stream_channels` 基于完整的 state channels，用于 `stream_mode="values"` 时输出全量状态。

## 6.5 第四步：创建 CompiledStateGraph

接下来是核心步骤——创建 `CompiledStateGraph` 实例：

```python
# langgraph/graph/state.py - compile()

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
    auto_validate=False,   # 先不验证，等填充完节点和边后再验证
    debug=debug,
    store=store,
    cache=cache,
    name=name or "LangGraph",
)
```

关键细节：channels 合并了用户定义的 channels、managed values 和一个 `START: EphemeralValue(self.input_schema)` channel（单步有效，下一步自动清空）。`input_channels` 固定为 `START`。`auto_validate=False` 因为节点和边尚未挂载。`nodes={}` 将在 `attach_node` 阶段填充。

## 6.6 CompiledStateGraph vs Pregel 继承关系

`CompiledStateGraph` 继承自 `Pregel`：

```python
# langgraph/graph/state.py

class CompiledStateGraph(
    Pregel[StateT, ContextT, InputT, OutputT],
    Generic[StateT, ContextT, InputT, OutputT],
):
    builder: StateGraph[StateT, ContextT, InputT, OutputT]
    schema_to_mapper: dict[type[Any], Callable[[Any], Any] | None]
    _output_mapper: Callable[[Any], Any] | None
    _state_mapper: Callable[[Any], Any] | None
```

`CompiledStateGraph` 在 `Pregel` 基础上增加了 `builder`（原始 `StateGraph` 引用）、`schema_to_mapper`（schema -> mapper 缓存）和 `_output_mapper`/`_state_mapper`（将 channel dict 转回 Pydantic model 或 dataclass）。`Pregel.__init__` 还会自动注入 `TASKS` channel（`Topic(Send, accumulate=False)`），这是 `Send` 机制的底层支撑。

## 6.7 第五步：挂载节点——attach_node

`compile()` 方法逐一调用 `attach_node` 将声明式的 `StateNodeSpec` 转化为 Pregel 运行时的 `PregelNode`：

```python
# langgraph/graph/state.py - compile()

compiled.attach_node(START, None)
for key, node in self.nodes.items():
    compiled.attach_node(key, node)
```

### PregelNode 的结构

`PregelNode` 是 Pregel 运行时的核心数据结构，定义于 `pregel/_read.py`：

```python
# langgraph/pregel/_read.py

class PregelNode:
    channels: str | list[str]
    """读取的 channel。str 表示单 channel 输入，list 表示多 channel 组成 dict 输入。"""

    triggers: list[str]
    """触发 channel 列表。任一被写入即触发本节点执行。"""

    mapper: Callable[[Any], Any] | None
    """输入转换函数，将 channel dict 转为 schema 类（如 Pydantic model）。"""

    writers: list[Runnable]
    """节点执行后的写入器列表，负责将输出写入 channel。"""

    bound: Runnable[Any, Any]
    """节点的核心逻辑。"""

    retry_policy: Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    tags: Sequence[str] | None
    metadata: Mapping[str, Any] | None
    subgraphs: Sequence[PregelProtocol]
```

`PregelNode` 不是一个 Runnable，它是一个**容器**——Pregel 从中提取信息来构建 `PregelExecutableTask`。它的 `node` 属性将 `bound` 和 `writers` 组合为一个可执行链：

```python
# langgraph/pregel/_read.py

@cached_property
def node(self) -> Runnable[Any, Any] | None:
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

### attach_node 的实现

对于 `START` 节点，创建一个仅转发输入的 `PregelNode`：

```python
# langgraph/graph/state.py - attach_node

if key == START:
    self.nodes[key] = PregelNode(
        tags=[TAG_HIDDEN],
        triggers=[START],
        channels=START,
        writers=[ChannelWrite(write_entries)],
    )
```

对于用户节点，过程更为复杂：

```python
elif node is not None:
    input_schema = node.input_schema if node else self.builder.state_schema
    input_channels = list(self.builder.schemas[input_schema])
    is_single_input = len(input_channels) == 1 and "__root__" in input_channels

    # 获取或创建 mapper（将 dict 转为 Pydantic model 等）
    if input_schema in self.schema_to_mapper:
        mapper = self.schema_to_mapper[input_schema]
    else:
        mapper = _pick_mapper(input_channels, input_schema)
        self.schema_to_mapper[input_schema] = mapper

    # 为该节点创建 trigger channel
    branch_channel = _CHANNEL_BRANCH_TO.format(key)  # "branch:to:{node_name}"
    self.channels[branch_channel] = (
        LastValueAfterFinish(Any)  # defer 节点用
        if node.defer
        else EphemeralValue(Any, guard=False)  # 普通节点用临时 channel
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

这里有几个核心设计：

**1. Trigger channel：** 每个用户节点都有一个专属的 trigger channel，命名为 `branch:to:{node_name}`。该 channel 使用 `EphemeralValue`——只在写入当步有效。当某条边要触发节点 X 时，就往 `branch:to:X` 写入一个值。

**2. Channels（读取源）：** 决定了节点的输入来自哪些 state 字段。如果节点指定了 `input_schema`，只读取该 schema 包含的字段。

**3. Writers（写入器）：** 包含一个 `ChannelWrite`，内含两个 `ChannelWriteTupleEntry`——一个负责解析节点输出为 state 更新（`_get_updates`），另一个负责解析 `Command` 中的路由控制（`_control_branch`）。

### _get_updates 闭包

每个节点的 writer 中包含一个 `_get_updates` 闭包，负责将节点返回值转为 `(key, value)` 对。它能处理 dict、`Command`、`Command` 列表以及 Pydantic model 等多种返回形式。`output_keys` 在闭包创建时捕获，只有属于已知 channel 的 key 才会被写入。发往父图的 `Command`（`graph == Command.PARENT`）会返回 `None`，不写入本地 state。

## 6.8 第六步：挂载边——attach_edge

```python
# langgraph/graph/state.py - compile()

for start, end in self.edges:
    compiled.attach_edge(start, end)

for starts, end in self.waiting_edges:
    compiled.attach_edge(starts, end)
```

`attach_edge` 的实现区分单起点和多起点：

```python
# langgraph/graph/state.py

def attach_edge(self, starts: str | Sequence[str], end: str) -> None:
    if isinstance(starts, str):
        if end != END:
            # 在起始节点的 writers 中追加一个 ChannelWrite
            # 写入目标节点的 trigger channel
            self.nodes[starts].writers.append(
                ChannelWrite(
                    (ChannelWriteEntry(_CHANNEL_BRANCH_TO.format(end), None),)
                )
            )
    elif end != END:
        # fan-in：创建 NamedBarrierValue channel
        channel_name = f"join:{'+'.join(starts)}:{end}"
        if self.builder.nodes[end].defer:
            self.channels[channel_name] = NamedBarrierValueAfterFinish(
                str, set(starts)
            )
        else:
            self.channels[channel_name] = NamedBarrierValue(str, set(starts))
        # 目标节点订阅这个 barrier channel
        self.nodes[end].triggers.append(channel_name)
        # 每个起始节点完成时写入 barrier
        for start in starts:
            self.nodes[start].writers.append(
                ChannelWrite((ChannelWriteEntry(channel_name, start),))
            )
```

**单起点边的机制：** 当节点 A 执行完毕后，它的 writers 会将 `None` 写入 `branch:to:B`，从而触发节点 B。

**Fan-in 边的机制：** 使用 `NamedBarrierValue` channel。这是一个特殊 channel，它收集多个命名写入（每个起始节点写入自己的名字），当所有预期的名字都到齐后才产生一个有效值，触发目标节点。

注意，指向 `END` 的边**不生成任何 channel 写入**。`END` 不是一个真正的节点——当没有任何节点被触发时，图自然结束。

## 6.9 第七步：挂载条件分支——attach_branch

```python
# langgraph/graph/state.py - compile()

for start, branches in self.branches.items():
    for name, branch in branches.items():
        compiled.attach_branch(start, name, branch)
```

`attach_branch` 的实现：

```python
# langgraph/graph/state.py

def attach_branch(
    self, start: str, name: str, branch: BranchSpec, *, with_reader: bool = True
) -> None:
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
        return writes

    if with_reader:
        schema = branch.input_schema or (
            self.builder.nodes[start].input_schema
            if start in self.builder.nodes
            else self.builder.state_schema
        )
        channels = list(self.builder.schemas[schema])
        mapper = _pick_mapper(channels, schema)
        reader = partial(
            ChannelRead.do_read,
            select=channels[0] if channels == ["__root__"] else channels,
            fresh=True,
            mapper=mapper,
        )
    else:
        reader = None

    self.nodes[start].writers.append(branch.run(get_writes, reader))
```

这里的关键是 `branch.run(get_writes, reader)`——它将 `BranchSpec` 转化为一个 `RunnableCallable`，作为起始节点的 writer 追加到 `writers` 列表中。

`reader` 在路由函数执行前重新读取最新 state（`fresh=True`），`get_writes` 将路由结果转化为 `ChannelWriteEntry` 写入 trigger channel。

## 6.10 第八步：最终验证

编译的最后一步调用 `Pregel.validate()`：

```python
# langgraph/graph/state.py - compile() 最后一行
return compiled.validate()
```

`Pregel.validate()` 委托给 `validate_graph` 函数：

```python
# langgraph/pregel/main.py

def validate(self) -> Self:
    validate_graph(
        self.nodes,
        {k: v for k, v in self.channels.items() if isinstance(v, BaseChannel)},
        {k: v for k, v in self.channels.items() if not isinstance(v, BaseChannel)},
        self.input_channels,
        self.output_channels,
        self.stream_channels,
        self.interrupt_after_nodes,
        self.interrupt_before_nodes,
    )
    self.trigger_to_nodes = _trigger_to_nodes(self.nodes)
    return self
```

`validate_graph`（`pregel/_validate.py`）执行 Pregel 层面的验证，按顺序检查：

1. **保留名检查**：channel 名、managed 名、节点名不能使用 `RESERVED` 中的保留字。
2. **channel 引用完整性**：每个 `PregelNode` 读取的 channels 和 triggers 必须在 `channels` 或 `managed` 字典中存在。
3. **输入 channel 检查**：input channel 必须存在且被至少一个节点订阅。
4. **输出 channel 检查**：output channels 和 stream channels 必须存在。
5. **interrupt 节点检查**：interrupt_before 和 interrupt_after 指定的节点必须存在。

验证完成后，`_trigger_to_nodes` 构建一个 trigger channel 到节点名的反向映射，供运行时快速查找。

## 6.11 编译产物总结

经过完整的编译流程，我们得到一个 `CompiledStateGraph` 实例，它包含：

| 组件 | 来源 | 说明 |
|------|------|------|
| `nodes: dict[str, PregelNode]` | `attach_node` | 每个用户节点 + START 节点 |
| `channels: dict[str, BaseChannel]` | State schema + `attach_edge` | state channels + trigger channels + barrier channels + TASKS channel |
| `input_channels` | 固定为 `START` | 输入 channel |
| `output_channels` | output schema | 输出 channel |
| `stream_channels` | state schema | 流式输出 channel |
| `checkpointer` | 用户传入 | 持久化保存器 |
| `trigger_to_nodes` | `_trigger_to_nodes` | trigger -> node 反向索引 |

以 `StateGraph(State).add_node("a", ...).add_node("b", ...).add_edge(START, "a").add_edge("a", "b").add_edge("b", END)` 为例，编译后的 channels 字典包含：`count`（`LastValue(int)`）、`__start__`（`EphemeralValue`）、`branch:to:a`（`EphemeralValue`）、`branch:to:b`（`EphemeralValue`）、`__pregel_tasks`（`Topic(Send)`）。节点 `__start__` 的 writers 写入 `branch:to:a`，节点 `a` 的 writers 写入 `branch:to:b`。Pregel 运行时通过检查哪些 trigger channel 被更新来决定下一步执行哪些节点。

## 6.12 Pregel 的执行模型回顾

编译产物最终由 `Pregel` 的执行引擎驱动。`Pregel` 遵循 **Bulk Synchronous Parallel（BSP）** 模型：每一步包含 Plan（确定就绪节点）、Execution（并行执行）、Update（写入 channel）三个阶段，循环直到没有节点被触发或达到 `recursion_limit`。

编译产生的 `PregelNode` 在运行时被转化为 `PregelExecutableTask`——一个 frozen dataclass，包含 `name`、`input`、`proc`（可执行 Runnable）、`writes`（写入缓冲区 deque）、`config`、`triggers`、`retry_policy`、`cache_key`、`id` 等字段。它是 Pregel 运行循环中的最小执行单元。

## 本章要点

1. **compile() 是声明式 API 到运行时 API 的桥梁**。`StateGraph` 是用户友好的声明层，`CompiledStateGraph`（继承 `Pregel`）是可执行的运行层。

2. **验证分两层**：`StateGraph.validate()` 检查拓扑合法性（节点存在性、入口点），`Pregel.validate()` 通过 `validate_graph` 检查 channel 引用完整性。LangGraph 有意**不做循环检测**，因为循环是其核心特性。

3. **PregelNode 是编译的核心产物**。它不是 Runnable，而是一个容器，封装了 `triggers`（触发条件）、`channels`（输入源）、`bound`（核心逻辑）和 `writers`（输出处理）。

4. **边的本质是 channel 写入**。静态边通过在源节点的 writers 中追加 `ChannelWriteEntry` 来触发目标节点。条件边通过追加 `BranchSpec.run()` 生成的 Runnable 来实现动态路由。

5. **每个用户节点有一个专属的 trigger channel**（`branch:to:{name}`），使用 `EphemeralValue` 实现单步触发语义。Fan-in 边通过 `NamedBarrierValue` channel 实现多源同步。

6. **CompiledStateGraph 在 Pregel 之上**增加了 `builder` 引用和 schema mapper，提供 JSON schema 导出和 state 类型转换能力。
