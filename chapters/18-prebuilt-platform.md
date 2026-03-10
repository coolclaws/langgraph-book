# 第 18 章 预构建 Agent、RemoteGraph、CLI 与 SDK

前面十七章我们从零拆解了 LangGraph 的每一层内核——从 Channel 到 Pregel、从 Checkpoint 到 Store、从 Stream 到 Interrupt。本章转向"开箱即用"层：LangGraph 提供的预构建组件（`prebuilt`）、分布式远程调用（`RemoteGraph`）、命令行工具（`langgraph` CLI）以及 Python SDK。这些上层设施屏蔽了底层细节，让使用者能以最少代码搭建可上线的 Agent 系统。

---

## 18.1 create_react_agent：ReAct 模式的内置实现

### 18.1.1 函数签名

`create_react_agent` 定义在 `libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py`，其核心签名如下：

```python
def create_react_agent(
    model: str | LanguageModelLike | Callable[..., BaseChatModel],
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | ToolNode,
    *,
    prompt: Prompt | None = None,
    response_format: StructuredResponseSchema | tuple[str, StructuredResponseSchema] | None = None,
    pre_model_hook: RunnableLike | None = None,
    post_model_hook: RunnableLike | None = None,
    state_schema: StateSchemaType | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
    name: str | None = None,
) -> CompiledStateGraph:
```

返回值是 `CompiledStateGraph`——也就是说，它在内部完成了 `StateGraph` 的构建、节点注册、边连接和编译。

### 18.1.2 内部图结构

`create_react_agent` 构建的图包含以下核心节点与边：

```
              ┌──────────────────┐
              │     START        │
              └───────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │  pre_model_hook?    │  ← 可选的前置钩子（消息裁剪/摘要）
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │      agent          │  ← 调用 LLM（prompt + model）
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │  post_model_hook?   │  ← 可选的后置钩子（人工审核/guardrail）
           └──────────┬──────────┘
                      │
              tools_condition
              ╱             ╲
     有 tool_calls      无 tool_calls
        │                    │
  ┌─────▼─────┐        ┌────▼────┐
  │   tools    │        │   END   │
  └─────┬─────┘        └─────────┘
        │
        └──────→ 回到 agent 节点（循环）
```

关键路由函数是 `tools_condition`，它检查最后一条 `AIMessage` 是否包含 `tool_calls`：

```python
def tools_condition(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
```

如果 `AIMessage.tool_calls` 非空，返回 `"tools"`，否则返回 `"__end__"`。这形成了 ReAct 的 "推理-行动-观察" 循环。

### 18.1.3 AgentState 与 remaining_steps

默认的状态 schema `AgentState` 只有两个字段：

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]
```

`remaining_steps` 是一个 `ManagedValue`，由 `RemainingStepsManager` 计算为 `recursion_limit - total_steps_taken`。当 `remaining_steps < 2` 且仍有 `tool_calls` 时，agent 会返回一条带有 "Sorry, need more steps to process this request." 内容的 `AIMessage`，而不会抛出 `GraphRecursionError`。这一机制使得递归耗尽时的行为更可预测。

### 18.1.4 Prompt 处理

`prompt` 参数支持四种形式，统一由 `_get_prompt_runnable` 转换为 `Runnable`：

| 类型 | 行为 |
|------|------|
| `str` | 包装为 `SystemMessage`，拼在 `messages` 前面 |
| `SystemMessage` | 直接拼在 `messages` 前面 |
| `Callable` | 接收完整 state，返回 `LanguageModelInput` |
| `Runnable` | 直接使用 |

### 18.1.5 动态模型选择

`model` 参数支持传入 `Callable[[StateSchema, Runtime[ContextT]], BaseChatModel]`，在每次调用 agent 节点时根据运行时上下文选择不同模型。这对于根据对话复杂度动态切换模型大小的场景非常实用。

### 18.1.6 response_format

当指定 `response_format` 时，图在工具循环结束后会额外调用 `model.with_structured_output(schema)` 来生成结构化输出，结果存储在 `state["structured_response"]` 中。

---

## 18.2 ToolNode：工具执行层

### 18.2.1 核心职责

`ToolNode`（`libs/prebuilt/langgraph/prebuilt/tool_node.py`）继承自 `RunnableCallable`，是工具执行的核心节点。它处理三种输入格式：

1. **Graph state**（字典，包含 `messages` 键）
2. **消息列表**（`[AIMessage(..., tool_calls=[...])]`）
3. **直接 tool_call 列表**（`[{"name": "tool", "args": {...}, "id": "1", "type": "tool_call"}]`）

### 18.2.2 并行执行与错误处理

`ToolNode` 使用 `get_executor_for_config` 获取线程池，将多个 `tool_call` 并行调度。错误处理通过 `handle_tool_errors` 参数配置，支持多种策略：

```python
handle_tool_errors:
    True          → 捕获所有错误，返回包含错误详情的 ToolMessage
    str           → 使用自定义错误信息字符串
    type[Exception] → 仅捕获特定异常类型
    Callable      → 自定义错误处理函数
    False         → 不捕获，异常直接传播
```

默认行为由 `_default_handle_tool_errors` 实现：捕获 `ToolInvocationError`（参数验证错误），但让工具执行时的其他异常继续上抛。

错误消息模板定义在模块顶部：

```python
INVALID_TOOL_NAME_ERROR_TEMPLATE = (
    "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]."
)
TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."
```

### 18.2.3 InjectedState 与 InjectedStore

工具函数可以通过 `Annotated` 类型注解请求注入额外上下文：

- **`InjectedState`**：让工具访问当前图状态
- **`InjectedStore`**：让工具访问持久化 Store

这些注入参数在 `ToolNode` 内部被识别并自动填充，LLM 不会看到这些参数（它们被标记为 `InjectedToolArg`，不会出现在 schema 中）。

### 18.2.4 ToolCallRequest 与拦截器

`ToolCallRequest` 是一个 dataclass，封装了工具调用的完整上下文：

```python
@dataclass
class ToolCallRequest:
    tool_call: ToolCall        # 来自 LLM 的工具调用（name, args, id）
    tool: BaseTool | None      # 对应的工具实例
    state: Any                 # 当前图状态
    runtime: ToolRuntime       # 运行时上下文
```

`ToolCallWrapper` 类型定义了拦截器的签名：

```python
ToolCallWrapper = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command]],
    ToolMessage | Command,
]
```

拦截器可以修改请求、实现重试逻辑、缓存结果、或完全短路执行。`ToolCallRequest` 通过 `override()` 方法创建新实例，遵循不可变模式。

### 18.2.5 Command 工具

工具可以返回 `Command` 对象来触发状态更新和图导航，而不仅仅是返回文本结果。这使得工具能够实现复杂的控制流，如跳转到特定节点或向其他节点发送消息。

---

## 18.3 ValidationNode：输出验证

`ValidationNode`（`libs/prebuilt/langgraph/prebuilt/tool_validator.py`）用于验证 LLM 的 `tool_calls` 是否符合指定的 Pydantic schema。它不执行工具，只做验证——适用于结构化信息提取场景。

### 18.3.1 工作原理

```python
class ValidationNode(RunnableCallable):
    def __init__(
        self,
        schemas: Sequence[BaseTool | type[BaseModel] | Callable],
        *,
        format_error: Callable[..., str] | None = None,
        name: str = "validation",
    ) -> None:
```

初始化时接收 schema 列表，支持三种来源：

| 来源 | 处理方式 |
|------|----------|
| `BaseTool` | 提取 `args_schema` |
| `type[BaseModel]` | 直接使用 |
| `Callable` | 通过 `create_schema_from_function` 推导 |

### 18.3.2 验证流程

对每个 `tool_call`，`ValidationNode` 调用 `schema.model_validate(call["args"])`：

- 成功：返回 `ToolMessage(content=output.model_dump_json())`
- 失败：返回 `ToolMessage(content=error_message, additional_kwargs={"is_error": True})`

典型用法是在 `model` → `validation` 之间构建一个重试循环，当验证失败时将错误反馈给 LLM 重新生成。

---

## 18.4 RemoteGraph：分布式图执行

### 18.4.1 设计定位

`RemoteGraph`（`libs/langgraph/langgraph/pregel/remote.py`）实现了 `PregelProtocol` 接口，使远程部署的 LangGraph 服务可以像本地 `CompiledGraph` 一样被调用。它可以直接作为另一个图的节点使用：

```python
remote = RemoteGraph("my-assistant", url="http://remote-server:8123")
builder.add_node("remote_agent", remote)
```

### 18.4.2 初始化

```python
class RemoteGraph(PregelProtocol):
    def __init__(
        self,
        assistant_id: str,
        /,
        *,
        url: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        client: LangGraphClient | None = None,
        sync_client: SyncLangGraphClient | None = None,
        config: RunnableConfig | None = None,
        name: str | None = None,
        distributed_tracing: bool = False,
    ):
```

`RemoteGraph` 内部持有异步 `LangGraphClient` 和同步 `SyncLangGraphClient`，通过 LangGraph Server API 进行 HTTP 通信。`assistant_id` 标识远程部署的图。

### 18.4.3 配置传递与清理

`RemoteGraph` 在传递 config 到远端时会过滤掉仅在本地有意义的配置键：

```python
_CONF_DROPLIST = frozenset((
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_TASK_ID,
))
```

`_sanitize_config_value` 函数递归清理 config 值，确保只包含可序列化的原始类型（`str`、`int`、`float`、`bool`、`UUID`）。

### 18.4.4 与本地图的互操作

由于 `RemoteGraph` 实现了 `PregelProtocol`，它支持 `invoke`、`stream`、`get_state`、`update_state` 等全部标准操作，因此可以：

- 作为子图嵌入本地图
- 跨服务组合多个 Agent
- 通过 `distributed_tracing=True` 启用 LangSmith 分布式追踪

---

## 18.5 LangGraph CLI

LangGraph CLI（`libs/cli/langgraph_cli/cli.py`）是基于 `click` 构建的命令行工具，提供以下核心命令：

### 18.5.1 命令总览

| 命令 | 用途 |
|------|------|
| `langgraph new` | 从模板创建新项目 |
| `langgraph dev` | 启动本地开发服务器（热重载） |
| `langgraph build` | 构建 Docker 镜像 |
| `langgraph up` | 启动 Docker 容器运行服务 |
| `langgraph dockerfile` | 生成 Dockerfile |

### 18.5.2 配置文件

CLI 读取项目根目录的 `langgraph.json` 配置文件（常量 `DEFAULT_CONFIG`），其中可以指定：

- 图的入口点和依赖
- 环境变量（支持 `.env` 文件引用）
- Python 依赖和系统包

环境变量处理由 `_parse_env_from_config` 完成，支持字典形式和 `.env` 文件路径两种配置方式。

### 18.5.3 保留环境变量

CLI 定义了 `RESERVED_ENV_VARS` 集合，包含 LangSmith/LangChain 平台相关的环境变量（如 `LANGCHAIN_API_KEY`、`POSTGRES_URI`、`REDIS_URI` 等），这些变量由平台自动注入，用户不应手动覆盖。

### 18.5.4 dev 命令

`langgraph dev` 启动一个带热重载的本地开发服务器，在默认端口 `DEFAULT_PORT` 上监听。它使用 `Runner` 类管理子进程，并集成了 `HostBackendClient` 进行本地后端通信。

### 18.5.5 build 与 up

`langgraph build` 利用 Docker 构建生产镜像，`langgraph up` 则启动容器。两者都通过 `DockerCapabilities` 检测当前系统的 Docker 环境能力。

---

## 18.6 Python SDK

### 18.6.1 架构总览

LangGraph SDK（`libs/sdk-py/langgraph_sdk/client.py`）提供了对 LangGraph Server API 的完整客户端封装。有两个入口工厂函数：

```python
from langgraph_sdk import get_client, get_sync_client

# 异步客户端
client = get_client(url="http://localhost:2024")

# 同步客户端
sync_client = get_sync_client(url="http://localhost:2024")
```

### 18.6.2 LangGraphClient 结构

```python
class LangGraphClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.http = HttpClient(client)
        self.assistants = AssistantsClient(self.http)
        self.threads = ThreadsClient(self.http)
        self.runs = RunsClient(self.http)
        self.crons = CronClient(self.http)
        self.store = StoreClient(self.http)
```

五个子客户端分别负责不同的资源域：

| 子客户端 | 职责 |
|----------|------|
| `AssistantsClient` | 管理 Assistant（图的版本化配置） |
| `ThreadsClient` | 管理 Thread（对话线程，关联 checkpoint） |
| `RunsClient` | 管理 Run（单次图调用，支持 `wait`/`stream`） |
| `CronClient` | 管理定时任务 |
| `StoreClient` | 管理持久化 Store 数据 |

同步版本镜像了相同的结构：`SyncLangGraphClient` 包含 `SyncAssistantsClient`、`SyncThreadsClient`、`SyncRunsClient` 等。

### 18.6.3 API Key 解析

SDK 支持从环境变量自动加载 API Key，按以下优先级：

1. `LANGGRAPH_API_KEY`
2. `LANGSMITH_API_KEY`
3. `LANGCHAIN_API_KEY`

也可以显式传入 `api_key=None` 跳过环境变量加载。

### 18.6.4 进程内连接

当 `url=None` 时，SDK 尝试通过 ASGI transport 建立进程内连接，这在 LangGraph Server 内部的 agent 间通信中非常有用——无需 HTTP 网络开销：

```python
client = get_client(url=None)
# 在 agent 节点内直接调用另一个 graph
result = await client.runs.wait(
    thread_id=None,
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Foo"}]},
)
```

### 18.6.5 超时配置

默认超时设置为：

```python
httpx.Timeout(connect=5, read=300, write=300, pool=5)
```

`read` 和 `write` 各 5 分钟，适应 LLM 调用可能的长延迟。

---

## 18.7 各组件间的协作关系

```
用户代码
  │
  ├─ create_react_agent() ──→ CompiledStateGraph
  │    ├─ agent 节点 (LLM)
  │    ├─ ToolNode (tools)
  │    └─ tools_condition (路由)
  │
  ├─ RemoteGraph ──HTTP──→ LangGraph Server
  │                            │
  │                    langgraph CLI 部署
  │                    (dev / build / up)
  │
  └─ LangGraphClient ──HTTP──→ LangGraph Server
       ├─ .assistants
       ├─ .threads
       ├─ .runs
       ├─ .crons
       └─ .store
```

`create_react_agent` 是构建时抽象，`RemoteGraph` 和 `LangGraphClient` 是运行时抽象，`CLI` 是部署时工具。三者共同覆盖了从开发到上线的完整生命周期。

---

## 本章要点

1. **create_react_agent** 在内部构建了一个 `StateGraph`，包含 `agent`（LLM 节点）和 `tools`（ToolNode）两个核心节点，由 `tools_condition` 路由形成 ReAct 循环。它支持动态模型选择、结构化输出、前置/后置钩子等高级特性。

2. **ToolNode** 是工具执行的核心抽象，支持并行执行、多种错误处理策略、`InjectedState`/`InjectedStore` 注入，以及通过 `ToolCallWrapper` 实现的拦截器模式。工具可以返回 `Command` 来实现复杂控制流。

3. **ValidationNode** 专注于 schema 验证而非工具执行，适合结构化信息提取场景，通过验证-重试循环引导 LLM 输出合规结果。

4. **RemoteGraph** 实现了 `PregelProtocol`，将远程 LangGraph 服务包装为可直接嵌入本地图的节点。它在传递配置时会清理仅限本地的配置键，确保跨网络的正确性。

5. **LangGraph CLI** 提供了 `new`、`dev`、`build`、`up`、`dockerfile` 五个核心命令，覆盖从项目创建到容器化部署的完整流程。

6. **Python SDK** 通过 `LangGraphClient` 提供了 `AssistantsClient`、`ThreadsClient`、`RunsClient`、`CronClient`、`StoreClient` 五个子客户端，支持异步和同步两种模式，以及进程内零开销连接。

7. 这三层（预构建组件 / CLI 部署 / SDK 远程调用）共同构成了 LangGraph 的"平台化"能力，让使用者能够从单机原型快速过渡到分布式生产环境。
