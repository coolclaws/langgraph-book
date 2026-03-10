# 附录 C：名词解释

本附录按字母顺序收录 LangGraph 源码中的核心术语，标注其定义位置和相关章节。

---

## A

**Agent**
在 LangGraph 语境中，Agent 是一个通过 LLM 驱动的循环决策系统：LLM 判断是否需要调用工具，调用后将结果反馈给 LLM 继续推理，直到得出最终答案。`create_react_agent` 是其预构建实现。见第 18 章。

**AgentState**
Agent 的默认状态 schema，定义在 `prebuilt/chat_agent_executor.py`，包含 `messages`（消息列表，使用 `add_messages` 作为 reducer）和 `remaining_steps`（剩余步数，ManagedValue）两个字段。见第 18 章。

**AnyValue**
Channel 类型之一，定义在 `langgraph/channels/any_value.py`。接受任意单次更新，如果同一 superstep 内收到多次更新则抛出 `InvalidUpdateError`。见第 4 章。

**AssistantsClient**
SDK 中管理 Assistant 资源的客户端类，定义在 `sdk-py/langgraph_sdk/_async/assistants.py`。Assistant 是图的版本化配置。见第 18 章。

## B

**BaseChannel**
所有 Channel 的抽象基类，定义在 `langgraph/channels/base.py`，泛型参数为 `Generic[Value, Update, Checkpoint]`。定义了 `update`、`get`、`checkpoint`、`from_checkpoint` 四个核心接口。见第 4 章。

**BaseCheckpointSaver**
Checkpoint 存储的抽象基类，定义在 `checkpoint/langgraph/checkpoint/base/__init__.py`。子类需实现 `get_tuple`、`put`、`list`、`put_writes` 方法。见第 11、12 章。

**BaseStore**
跨 Thread 持久化存储的抽象基类，定义在 `checkpoint/langgraph/store/base/__init__.py`。提供 `get`、`put`、`search`、`delete`、`list_namespaces`、`batch` 等操作。见第 13 章。

**BinaryOperatorAggregate**
Channel 类型之一，定义在 `langgraph/channels/binop.py`。通过用户提供的二元运算符（reducer 函数）将更新值累积到当前值上，如 `operator.add` 用于列表拼接、`add_messages` 用于消息合并。见第 4 章。

## C

**CachePolicy**
缓存策略配置，定义在 `langgraph/types.py`，`Generic[KeyFuncT]`。通过 `key` 函数计算缓存键，当节点的输入命中缓存时跳过执行，直接复用之前的结果。见第 10 章。

**Channel**
LangGraph Pregel 模型中节点间通信的抽象。每个 State 字段在运行时对应一个 Channel 实例，Channel 管理值的接收、聚合和检查点序列化。见第 4 章。

**Checkpoint**
图执行的快照数据结构，定义在 `checkpoint/base/__init__.py`，是一个 `TypedDict`，包含 `id`、`ts`、`channel_values`、`channel_versions`、`versions_seen`、`pending_sends` 等字段。每个 superstep 结束后产生一个新的 Checkpoint。见第 11 章。

**CheckpointMetadata**
附加在 Checkpoint 上的元数据，`TypedDict(total=False)`，记录 `source`（产生来源：`"input"`、`"loop"`、`"update"`）、`step`（步号）、`writes`（本步写入）、`parents`（父图 checkpoint 信息）。见第 11 章。

**CheckpointTuple**
`NamedTuple`，包含 `config`、`checkpoint`、`metadata`、`parent_config` 四个字段，是 `BaseCheckpointSaver` 读写的基本单元。见第 11 章。

**Command**
控制流指令，定义在 `langgraph/types.py`，`Generic[N]`。可以同时执行三种操作：通过 `update` 更新状态、通过 `goto` 导航到指定节点、通过 `resume` 恢复中断的执行。见第 15 章。

**CompiledStateGraph**
`StateGraph.compile()` 的返回类型，定义在 `langgraph/graph/state.py`。包含完整的 Pregel 运行时结构（PregelNode 映射、Channel 映射、Stream 配置等），实现 `PregelProtocol` 接口。见第 3、6 章。

## D

**DynamicBarrierValue**
动态屏障 Channel，运行时注册等待的生产者集合，当所有注册的生产者都写入后释放。见第 4 章。

## E

**END**
图的虚拟结束节点标识符，值为 `"__end__"`。当条件边返回 `END` 或所有后续节点执行完毕时，图结束执行。见第 3 章。

**Entrypoint**
函数式 API 中的入口点装饰器，将一个函数标记为图的起始节点。与 `task` 配合使用，提供比 `StateGraph` 更轻量的图定义方式。见第 5 章。

**EphemeralValue**
Channel 类型之一，定义在 `langgraph/channels/ephemeral_value.py`。值仅在当前 superstep 有效，下一个 superstep 开始时自动清空，且不参与 checkpoint 持久化。见第 4 章。

## F

**Fork**
Time Travel 的一种操作模式：从历史 Checkpoint 创建一个新的执行分支，原有的执行历史不受影响。通过向 `update_state` 传入历史 checkpoint 的 config 实现。见第 11 章。

## G

**GraphBubbleUp**
控制流异常的基类，定义在 `langgraph/errors.py`。`GraphInterrupt` 和 `ParentCommand` 都继承自此类。这类异常不表示错误，而是用于在图的执行栈中向上传播控制信号。见第 14 章。

**GraphInterrupt**
由 `interrupt()` 函数触发的异常，继承自 `GraphBubbleUp`，定义在 `langgraph/errors.py`。携带 `Interrupt` 数据向上冒泡，最终被 `PregelLoop` 捕获并保存到 checkpoint 中。见第 14 章。

**GraphRecursionError**
当执行步数超过 `recursion_limit` 时抛出的异常，继承自 `RecursionError`，定义在 `langgraph/errors.py`。见第 7 章。

## H

**Human-in-the-Loop**
人机协作模式，通过 `interrupt()` 暂停图执行、将控制权交给人类审核，然后通过 `Command(resume=...)` 恢复执行。实现方式包括 `interrupt_before`/`interrupt_after` 配置和显式 `interrupt()` 调用。见第 14 章。

## I

**InjectedState**
类型注解标记，用于工具函数参数。当工具被 `ToolNode` 执行时，标记为 `InjectedState` 的参数会自动注入当前图状态，且该参数不会出现在工具的 schema 中（LLM 不可见）。见第 18 章。

**InjectedStore**
类似 `InjectedState`，但注入的是 `BaseStore` 实例，让工具能够访问持久化存储。见第 18 章。

**Interrupt**
中断数据载体，定义在 `langgraph/types.py`。包含 `value`（传递给用户的数据）字段。`interrupt()` 函数创建 `Interrupt` 实例并抛出 `GraphInterrupt` 异常。见第 14 章。

**Item**
Store 中的数据条目，定义在 `checkpoint/langgraph/store/base/__init__.py`。包含 `namespace`（命名空间元组）、`key`（键）、`value`（JSON 可序列化的值）、`created_at`、`updated_at` 等字段。见第 13 章。

## L

**LangGraphClient**
异步顶层 SDK 客户端，定义在 `sdk-py/langgraph_sdk/_async/client.py`。包含 `assistants`、`threads`、`runs`、`crons`、`store` 五个子客户端。见第 18 章。

**LastValue**
最常用的 Channel 类型，定义在 `langgraph/channels/last_value.py`。只保留最后一个写入的值，多次写入时抛出 `InvalidUpdateError`。对应无 reducer 注解的 State 字段。见第 4 章。

## M

**ManagedValue**
运行时自动注入到 State 中的动态值，抽象基类定义在 `langgraph/managed/base.py`。不同于普通 State 字段，ManagedValue 不由用户显式更新，也不参与 checkpoint 持久化。`RemainingSteps` 是典型示例。见第 17 章。

**MessageGraph**
`StateGraph` 的便捷子类，定义在 `langgraph/graph/message.py`。状态自动设为 `Annotated[list, add_messages]`，适合纯消息驱动的对话场景。见第 2 章。

## N

**NamedBarrierValue**
命名屏障 Channel，定义在 `langgraph/channels/named_barrier_value.py`。初始化时指定一组生产者名称，只有当所有指定的生产者都写入后，Channel 才会释放（`get` 不再抛出 `EmptyChannelError`）。用于多节点同步。见第 4 章。

**Node**
图中的计算单元。用户通过 `StateGraph.add_node(name, fn)` 注册，编译后映射为 `PregelNode`。每个 Node 读取指定 Channel 的值、执行计算、将结果写入输出 Channel。见第 3、6 章。

## P

**ParentCommand**
子图向父图发送 Command 时使用的异常类型，定义在 `langgraph/errors.py`。继承自 `GraphBubbleUp`，携带 `Command` 数据向上冒泡直到被父图的 `PregelLoop` 捕获。见第 15、16 章。

**Pregel**
LangGraph 运行时的核心模型，灵感来自 Google 的 Pregel 图计算框架。节点通过 Channel 通信，执行以 superstep 为单位同步推进。定义在 `langgraph/pregel/__init__.py`。见第 6 章。

**PregelExecutableTask**
可执行任务，定义在 `langgraph/types.py`。是 `PregelLoop.tick()` 的调度单元，包含节点名称、输入数据、要执行的 Runnable、RunnableConfig 等信息。见第 7 章。

**PregelLoop**
执行引擎核心，定义在 `langgraph/pregel/_loop.py`。`tick()` 方法实现了一个 superstep：检查 Channel 状态 -> 确定可触发的节点 -> 创建 Task -> 执行 -> 收集输出 -> 更新 Channel -> checkpoint。见第 7 章。

**PregelNode**
Pregel 运行时的节点表示，定义在 `langgraph/pregel/_read.py`。封装了节点的 Runnable、输入 Channel 映射（triggers、channels）、输出 Channel 映射、writers 等信息。见第 6 章。

**PregelProtocol**
标准图操作协议，定义在 `langgraph/pregel/protocol.py`。规定了 `invoke`、`stream`、`get_state`、`update_state`、`get_graph` 等接口。`CompiledStateGraph` 和 `RemoteGraph` 都实现了此协议。见第 6、18 章。

## R

**Reducer**
State 字段上的聚合函数，通过 `Annotated[type, reducer_fn]` 注解。当多个节点在同一 superstep 写入同一字段时，reducer 决定如何合并更新。底层对应 `BinaryOperatorAggregate` Channel。见第 4 章。

**RemainingSteps**
`ManagedValue` 的一个实例，由 `RemainingStepsManager` 计算为 `recursion_limit - total_steps_taken`。当值小于 2 且 agent 仍有工具调用时，触发安全退出。见第 17、18 章。

**RemoteGraph**
远程图客户端，定义在 `langgraph/pregel/remote.py`，实现 `PregelProtocol`。通过 HTTP 调用部署在远端的 LangGraph Server，可以像本地图一样使用（包括作为子图嵌入）。见第 18 章。

**RetryPolicy**
重试策略配置，定义在 `langgraph/types.py`，`NamedTuple`。包含 `initial_interval`（初始间隔）、`backoff_factor`（退避因子）、`max_interval`（最大间隔）、`max_attempts`（最大重试次数）、`jitter`（随机抖动）、`retry_on`（可重试的异常判断函数）。见第 10 章。

**RunsClient**
SDK 中管理 Run 资源的客户端类。Run 代表一次图调用，支持 `create`（异步）、`wait`（同步等待结果）、`stream`（流式输出）等操作。见第 18 章。

## S

**Send**
fan-out 控制原语，定义在 `langgraph/types.py`。包含 `node`（目标节点名）和 `arg`（发送的数据）。条件边可以返回 `Send` 列表来向多个节点（或同一节点的多个实例）并行发送数据。见第 15 章。

**START**
图的虚拟起始节点标识符，值为 `"__start__"`。用户输入通过 START 节点进入图。见第 3 章。

**State**
图中所有节点共享的数据结构，由用户通过 `TypedDict` 或 `BaseModel` 定义。每个字段对应一个 Channel 实例。带 `Annotated` reducer 注解的字段使用 `BinaryOperatorAggregate`，否则使用 `LastValue`。见第 3、4 章。

**StateGraph**
图构建器，定义在 `langgraph/graph/state.py`。泛型类 `Generic[StateT, ContextT, InputT, OutputT]`。提供声明式 API 定义节点和边，通过 `compile()` 转换为 `CompiledStateGraph`。见第 3 章。

**StateSnapshot**
图状态的用户友好快照，定义在 `langgraph/types.py`，`NamedTuple`。包含 `values`（当前状态值）、`next`（下一步将执行的节点元组）、`config`、`metadata`、`tasks`（待执行任务列表）、`parent_config`（父 checkpoint 配置）。见第 11 章。

**Store**
跨 Thread 的持久化键值存储。不同于 Checkpoint（per-Thread 状态快照），Store 允许不同 Thread 之间共享和查询数据。见 `BaseStore`。见第 13 章。

**StreamMode**
流输出模式，控制 `stream()` 方法返回哪些类型的数据。常见模式包括 `"values"`（每步完整状态）、`"updates"`（每步增量更新）、`"debug"`（调试信息）、`"messages"`（LLM token 流）、`"custom"`（用户自定义流）。见第 8 章。

**StreamProtocol**
流输出的内部协议，定义在 `langgraph/pregel/protocol.py`。在 `PregelLoop` 执行过程中，通过 `StreamProtocol` 将中间数据写入到调用者可消费的流中。见第 8 章。

**Subgraph**
子图，即嵌入另一个图（父图）中作为节点执行的 `CompiledStateGraph`。子图拥有独立的状态和 Channel，通过输入/输出映射与父图交互。见第 16 章。

**Superstep**
Pregel 计算模型中的基本执行单位。在一个 superstep 中，所有被触发的节点并行执行、将输出写入 Channel、然后系统同步进入下一个 superstep。`PregelLoop.tick()` 执行一个 superstep。见第 6、7 章。

## T

**Task**
运行时的工作单元，有两个层面含义：`PregelExecutableTask` 是引擎内部的调度单元，`PregelTask` 是面向用户的任务描述符（出现在 `StateSnapshot.tasks` 中）。见第 7 章。

**Thread**
对话线程，对应一条独立的 Checkpoint 链。每个 Thread 有自己的 `thread_id`，所有 checkpoint 按时间顺序形成链表。不同 Thread 之间的状态完全隔离。见第 11、12 章。

**ThreadsClient**
SDK 中管理 Thread 资源的客户端类。支持 `create`、`get`、`search`、`get_state`、`update_state`、`get_history` 等操作。见第 18 章。

**TimeTravel**
时间旅行，通过加载历史 Checkpoint 回到图执行的某个过去时刻。支持两种模式：Replay（在同一 Thread 上重新执行）和 Fork（创建新分支）。见第 11 章。

**ToolNode**
工具执行节点，定义在 `prebuilt/langgraph/prebuilt/tool_node.py`。继承自 `RunnableCallable`，负责解析 `AIMessage.tool_calls`、分发工具调用、收集结果、处理错误。见第 18 章。

**Topic**
Channel 类型之一，定义在 `langgraph/channels/topic.py`。广播模式 Channel，在一个 superstep 中累积所有写入的值（作为列表），供下一个 superstep 的消费者读取，读取后清空。见第 4 章。

**tools_condition**
条件路由函数，定义在 `prebuilt/langgraph/prebuilt/tool_node.py`。检查最后一条 `AIMessage` 是否包含 `tool_calls`，返回 `"tools"` 或 `"__end__"`。是 ReAct 循环的关键路由逻辑。见第 18 章。

## U

**UntrackedValue**
Channel 类型之一，定义在 `langgraph/channels/untracked_value.py`。行为类似 `LastValue`，但不参与 checkpoint 版本追踪，Channel 的更新不会触发依赖节点的重新执行。见第 4 章。

## V

**ValidationNode**
输出验证节点，定义在 `prebuilt/langgraph/prebuilt/tool_validator.py`。使用 Pydantic schema 验证 `AIMessage.tool_calls` 的参数，返回包含验证结果或错误信息的 `ToolMessage`。见第 18 章。

**versions_seen**
`Checkpoint` 中的字段，类型为 `dict[str, dict[str, str]]`，记录每个节点上次看到的各 Channel 版本号。用于判断哪些节点需要在下一个 superstep 中被触发（即哪些节点的输入 Channel 有了新的更新）。见第 11 章。
