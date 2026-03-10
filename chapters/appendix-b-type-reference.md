# 附录 B：核心类型速查

本附录按功能分类列出 LangGraph 源码中最重要的类型定义，标注其源文件路径、父类/协议，以及简要说明。读者可将此作为阅读源码时的快速索引。

> **路径约定**：所有路径均相对于 LangGraph 源码仓库根目录，即 `libs/` 下的各子包。

---

## B.1 Graph 构建类型

| 类型 | 源文件 | 父类 | 说明 |
|------|--------|------|------|
| `StateGraph` | `langgraph/graph/state.py` | `Generic[StateT, ContextT, InputT, OutputT]` | 核心图构建器，提供 `add_node`、`add_edge`、`add_conditional_edges`、`compile` 等 API |
| `CompiledStateGraph` | `langgraph/graph/state.py` | （见源码，继承编译后的图基类） | `StateGraph.compile()` 的返回值，包含完整的运行时结构 |
| `MessageGraph` | `langgraph/graph/message.py` | `StateGraph` | 以消息列表为状态的便捷子类，内置 `add_messages` reducer |
| `START` | `langgraph/graph/__init__.py` | `str` 常量 | 图的虚拟起始节点标识符 `"__start__"` |
| `END` | `langgraph/graph/__init__.py` | `str` 常量 | 图的虚拟结束节点标识符 `"__end__"` |

---

## B.2 Channel 类型

| 类型 | 源文件 | 父类 | 说明 |
|------|--------|------|------|
| `BaseChannel` | `langgraph/channels/base.py` | `Generic[Value, Update, Checkpoint]`, `ABC` | 所有 Channel 的抽象基类，定义 `update`、`get`、`checkpoint`、`from_checkpoint` 接口 |
| `LastValue` | `langgraph/channels/last_value.py` | `BaseChannel[Value, Value, Value]` | 只保留最后一个值，对应无 reducer 的 State 字段 |
| `BinaryOperatorAggregate` | `langgraph/channels/binop.py` | `BaseChannel[Value, Value, Value]` | 用二元运算符（reducer）累积更新，如 `add_messages`、`operator.add` |
| `Topic` | `langgraph/channels/topic.py` | （见源码） | 广播 Channel，每个 superstep 累积所有更新，下一个 superstep 清空 |
| `EphemeralValue` | `langgraph/channels/ephemeral_value.py` | `BaseChannel[Value, Value, Value]` | 临时值，仅在当前 superstep 有效，不参与 checkpoint |
| `AnyValue` | `langgraph/channels/any_value.py` | `BaseChannel[Value, Value, Value]` | 接受任意单次更新的 Channel |
| `NamedBarrierValue` | `langgraph/channels/named_barrier_value.py` | `BaseChannel[Value, Value, set[Value]]` | 命名屏障，等待所有指定的生产者写入后才释放 |
| `UntrackedValue` | `langgraph/channels/untracked_value.py` | `BaseChannel[Value, Value, Value]` | 不被 checkpoint 追踪的 Channel，用于临时数据传递 |

---

## B.3 Pregel 类型

| 类型 | 源文件 | 父类 | 说明 |
|------|--------|------|------|
| `PregelNode` | `langgraph/pregel/_read.py` | （见源码） | Pregel 运行时的节点表示，封装了节点的 Runnable、输入/输出 Channel 映射、触发条件 |
| `PregelLoop` | `langgraph/pregel/_loop.py` | — | 执行引擎核心，实现 `tick()` 方法驱动 superstep 循环，管理 Channel 状态和 Task 调度 |
| `PregelExecutableTask` | `langgraph/types.py` | — | 可执行任务，包含节点名称、输入数据、Runnable、config 等，是 `PregelLoop.tick()` 的调度单元 |
| `PregelProtocol` | `langgraph/pregel/protocol.py` | — | 定义了 `invoke`、`stream`、`get_state` 等标准图操作接口，`CompiledStateGraph` 和 `RemoteGraph` 均实现此协议 |
| `StreamProtocol` | `langgraph/pregel/protocol.py` | — | 流输出协议，定义了运行时如何收集和分发 stream 数据 |

---

## B.4 Checkpoint 类型

| 类型 | 源文件 | 父类 | 说明 |
|------|--------|------|------|
| `Checkpoint` | `checkpoint/langgraph/checkpoint/base/__init__.py` | `TypedDict` | 检查点数据结构，包含 `v`（版本）、`id`、`ts`（时间戳）、`channel_values`、`channel_versions`、`versions_seen`、`pending_sends` |
| `CheckpointTuple` | `checkpoint/langgraph/checkpoint/base/__init__.py` | `NamedTuple` | 检查点元组，包含 `config`、`checkpoint`、`metadata`、`parent_config`，是 Saver 的读写单元 |
| `CheckpointMetadata` | `checkpoint/langgraph/checkpoint/base/__init__.py` | `TypedDict` (total=False) | 检查点元数据，记录 `source`（input/loop/update）、`step`、`writes`、`parents` |
| `BaseCheckpointSaver` | `checkpoint/langgraph/checkpoint/base/__init__.py` | `Generic[V]` | 检查点存储的抽象基类，定义 `get_tuple`、`put`、`list`、`put_writes` 接口 |

---

## B.5 Store 类型

| 类型 | 源文件 | 父类 | 说明 |
|------|--------|------|------|
| `BaseStore` | `checkpoint/langgraph/store/base/__init__.py` | `ABC` | Store 的抽象基类，定义 `get`、`put`、`search`、`delete`、`list_namespaces`、`batch` 等操作 |
| `Item` | `checkpoint/langgraph/store/base/__init__.py` | — | Store 中的数据条目，包含 `namespace`、`key`、`value`、`created_at`、`updated_at` |

---

## B.6 控制流类型

| 类型 | 源文件 | 父类 | 说明 |
|------|--------|------|------|
| `Command` | `langgraph/types.py` | `Generic[N]`, `ToolOutputMixin` | 控制流指令，可同时更新状态（`update`）、导航到指定节点（`goto`）、恢复中断（`resume`） |
| `Send` | `langgraph/types.py` | — | 向指定节点发送数据，实现 fan-out 并行。包含 `node`（目标节点名）和 `arg`（输入数据） |
| `Interrupt` | `langgraph/types.py` | — | 中断数据载体，包含 `value`（传递给用户的数据）和 `resumable` 标志 |
| `RetryPolicy` | `langgraph/types.py` | `NamedTuple` | 重试策略配置，包含 `initial_interval`、`backoff_factor`、`max_interval`、`max_attempts`、`jitter`、`retry_on` |
| `CachePolicy` | `langgraph/types.py` | `Generic[KeyFuncT]` | 缓存策略，通过 `key` 函数计算缓存键，避免重复执行 |
| `PregelTask` | `langgraph/types.py` | `NamedTuple` | 任务描述符（面向用户），包含 `id`、`name`、`path`、`error`、`interrupts`、`state` |
| `StateSnapshot` | `langgraph/types.py` | `NamedTuple` | 图状态快照，包含 `values`、`next`（待执行节点）、`config`、`metadata`、`tasks`、`parent_config` |
| `StreamMode` | `langgraph/types.py` | — | 流模式枚举/类型，支持 `"values"`、`"updates"`、`"debug"`、`"messages"`、`"custom"` 等模式 |
| `StreamPart` | `langgraph/types.py` | — | 流数据片段，包含 `topic`（stream mode）和 `data` |
| `GraphOutput` | `langgraph/types.py` | `Generic[OutputT]` | 图输出的包装类型 |

---

## B.7 错误类型

| 类型 | 源文件 | 父类 | 说明 |
|------|--------|------|------|
| `GraphRecursionError` | `langgraph/errors.py` | `RecursionError` | 达到 `recursion_limit` 时抛出 |
| `GraphBubbleUp` | `langgraph/errors.py` | `Exception` | 需要向上冒泡的控制流异常基类 |
| `GraphInterrupt` | `langgraph/errors.py` | `GraphBubbleUp` | `Interrupt` 触发的异常，携带中断数据 |
| `ParentCommand` | `langgraph/errors.py` | （见源码） | 子图向父图发送 Command 时使用的异常 |
| `RemoteException` | `langgraph/pregel/remote.py` | `Exception` | `RemoteGraph` 远程调用失败时抛出 |

---

## B.8 Prebuilt 类型

| 类型 | 源文件 | 父类 | 说明 |
|------|--------|------|------|
| `ToolNode` | `prebuilt/langgraph/prebuilt/tool_node.py` | `RunnableCallable` | 工具执行节点，支持并行调度、错误处理、状态注入 |
| `ValidationNode` | `prebuilt/langgraph/prebuilt/tool_validator.py` | `RunnableCallable` | Schema 验证节点，不执行工具，仅验证 tool_calls 参数 |
| `ToolCallRequest` | `prebuilt/langgraph/prebuilt/tool_node.py` | `dataclass` | 工具调用请求，包含 `tool_call`、`tool`、`state`、`runtime` |
| `ToolCallWrapper` | `prebuilt/langgraph/prebuilt/tool_node.py` | `Callable` 类型别名 | 工具调用拦截器签名 |
| `ToolCallWithContext` | `prebuilt/langgraph/prebuilt/tool_node.py` | `TypedDict` | 携带上下文的工具调用，用于 Send API 分发 |
| `RemoteGraph` | `langgraph/pregel/remote.py` | `PregelProtocol` | 远程图客户端，通过 HTTP 调用部署的 LangGraph 服务 |

---

## B.9 SDK 类型

| 类型 | 源文件 | 说明 |
|------|--------|------|
| `LangGraphClient` | `sdk-py/langgraph_sdk/_async/client.py` | 异步顶层客户端，包含 assistants/threads/runs/crons/store 子客户端 |
| `SyncLangGraphClient` | `sdk-py/langgraph_sdk/_sync/client.py` | 同步顶层客户端 |
| `AssistantsClient` | `sdk-py/langgraph_sdk/_async/assistants.py` | Assistant 资源管理 |
| `ThreadsClient` | `sdk-py/langgraph_sdk/_async/threads.py` | Thread 资源管理 |
| `RunsClient` | `sdk-py/langgraph_sdk/_async/runs.py` | Run 资源管理 |
| `CronClient` | `sdk-py/langgraph_sdk/_async/cron.py` | 定时任务管理 |
| `StoreClient` | `sdk-py/langgraph_sdk/_async/store.py` | 持久化存储管理 |

---

## B.10 Managed Value 类型

| 类型 | 源文件 | 父类 | 说明 |
|------|--------|------|------|
| `ManagedValue` | `langgraph/managed/base.py` | `ABC`, `Generic[V]` | Managed Value 的抽象基类，在运行时自动注入到 State 中 |
| `RemainingStepsManager` | `langgraph/managed/is_last_step.py` | `ManagedValue[int]` | 计算剩余可执行步数，用于 `remaining_steps` 字段 |

---

## B.11 使用提示

1. **类型查找**：遇到不熟悉的类型时，先在本表中定位源文件，然后直接阅读对应的类定义和 docstring。

2. **Generic 参数**：许多类型使用了 Python 泛型（如 `BaseChannel[Value, Update, Checkpoint]`），阅读时注意 TypeVar 的约束条件。

3. **TypedDict vs dataclass vs NamedTuple**：LangGraph 混合使用了三种数据容器：
   - `TypedDict`：用于可序列化的数据结构（如 `Checkpoint`、`CheckpointMetadata`）
   - `dataclass`：用于运行时对象（如 `ToolCallRequest`）
   - `NamedTuple`：用于不可变记录（如 `CheckpointTuple`、`PregelTask`、`StateSnapshot`）

4. **协议与实现**：`PregelProtocol` 是核心协议，`CompiledStateGraph` 和 `RemoteGraph` 都是它的实现者——这是本地与远程图可互换使用的基础。
