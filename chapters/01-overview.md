# 第 1 章 项目概览与设计哲学

LangGraph 是一个由 LangChain Inc. 开发的低层次（low-level）Agent 编排框架。它并非试图用高层抽象隐藏复杂性，而是提供一套精确的图计算原语，让开发者对 Agent 的每一步执行拥有完全的控制权。本章将从项目定位、设计哲学、核心价值和生态关系四个维度，帮助读者建立对 LangGraph 的全局认知。

---

## 1.1 LangGraph 的定位：低层次 Agent 编排框架

打开 LangGraph 仓库的 `README.md`，第一句话就明确了它的定位：

> Trusted by companies shaping the future of agents -- including Klarna, Replit,
> Elastic, and more -- LangGraph is a **low-level orchestration framework** for
> building, managing, and deploying long-running, stateful agents.

这里有三个关键词值得深入理解。

### 1.1.1 关键词一：Low-level（低层次）

在 `README.md` 中，项目团队进一步强调：

> LangGraph provides low-level supporting infrastructure for *any* long-running,
> stateful workflow or agent. LangGraph does **not** abstract prompts or
> architecture.

与许多 Agent 框架不同，LangGraph 不会替你决定 prompt 的写法，不会替你选择 Agent
架构模式（ReAct、Plan-and-Execute 还是其他），更不会用"一个函数调用搞定一切"的
方式隐藏底层细节。它提供的是**基础设施层**——状态管理、执行调度、持久化、中断与
恢复——而将架构决策权留给开发者。

这种设计选择意味着：

1. **灵活性最大化**：任何 Agent 架构都可以在 LangGraph 上实现
2. **可调试性**：开发者能够看到并控制每一步执行
3. **学习曲线较陡**：需要理解图计算模型、通道机制等底层概念
4. **没有"魔法"**：所有行为都是显式的、可预测的

### 1.1.2 关键词二：Stateful（有状态）

LangGraph 的 `pyproject.toml` 中，`description` 字段写道：

```toml
# 文件: libs/langgraph/pyproject.toml

[project]
name = "langgraph"
version = "1.1.0"
description = "Building stateful, multi-actor applications with LLMs"
requires-python = ">=3.10"
```

> **源码位置**：`libs/langgraph/pyproject.toml` 第 6-10 行

"Stateful, multi-actor"——有状态、多参与者。这正是 LangGraph 与简单的 Chain 或
Pipeline 框架的根本区别。一个 LangGraph 应用本质上是一个**有状态的图
（stateful graph）**，图中的节点（node）通过读写共享状态来协作完成任务。

状态的有状态性体现在多个层面：

- **步骤内状态**：同一次运行中，节点之间通过共享的 State 对象通信
- **步骤间状态**：通过 checkpoint 机制，状态在超级步之间持久化
- **会话间状态**：通过 `thread_id` 配置，不同会话可以拥有独立的状态链
- **运行间状态**：通过 `BaseStore` 接口，可以实现跨运行的长期记忆

### 1.1.3 关键词三：Long-running（长时间运行）

Agent 与传统的请求-响应模式有本质区别。一个 Agent 可能需要：

- 多次调用 LLM（思考-行动-观察循环）
- 等待外部 API 响应
- 等待人类审批
- 运行数分钟甚至数小时

这些场景要求框架能够：

1. **跨进程持久化**：进程崩溃后能恢复
2. **支持中断和恢复**：等待人类输入时不占用资源
3. **超时和重试**：优雅处理外部依赖的失败
4. **增量执行**：不需要从头重新运行整个工作流

LangGraph 通过 checkpoint 机制和 interrupt 机制完整支持了这些需求。


### 1.1.4 一个最小示例

在深入架构之前，先来看官方 README 中的最小可运行示例。这个示例虽然简单，但已经
体现了 LangGraph 的核心编程模型：

```python
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    text: str


def node_a(state: State) -> dict:
    return {"text": state["text"] + "a"}


def node_b(state: State) -> dict:
    return {"text": state["text"] + "b"}


graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")

print(graph.compile().invoke({"text": ""}))
# {'text': 'ab'}
```

> **源码位置**：`README.md` 第 28-52 行

这段代码展示了 LangGraph 编程模型的五个要素：

| 要素 | 示例中的体现 | 源码入口 |
|------|-------------|---------|
| **状态定义** | `class State(TypedDict)` | `libs/langgraph/langgraph/graph/state.py` |
| **节点函数** | `node_a`, `node_b` | 任何 `State -> Partial<State>` 的函数 |
| **图构建** | `StateGraph(State)` | `libs/langgraph/langgraph/graph/state.py` 的 `StateGraph` 类 |
| **边定义** | `add_edge(START, "node_a")` | 同上，`add_edge` 方法 |
| **编译与执行** | `graph.compile().invoke(...)` | `libs/langgraph/langgraph/pregel/main.py` 的 `Pregel` 类 |

注意节点函数的签名：`State -> dict`。每个节点接收完整的状态，返回状态的**部分
更新**（partial update）。这正是 `StateGraph` 文档中所说的 "The signature of
each node is `State -> Partial<State>`"。

这里的 `.compile()` 是一个至关重要的步骤——它将声明式的图定义转换为可执行的
`Pregel` 运行时。这个 Builder -> Compiler -> Runtime 的分层是 LangGraph
架构的核心模式，我们将在后续章节详细分析。


## 1.2 设计哲学：从 Google Pregel 到 LangGraph

### 1.2.1 Pregel 论文的启发

LangGraph 的命名和核心算法直接来源于 Google 在 2010 年发表的 Pregel 论文
（*Pregel: A System for Large-Scale Graph Processing*）。在项目 `README.md` 的
致谢部分，官方明确指出了它的学术渊源：

> LangGraph is inspired by [Pregel](https://research.google/pubs/pub37252/) and
> [Apache Beam](https://beam.apache.org/). The public interface draws inspiration
> from [NetworkX](https://networkx.org/documentation/latest/).

> **源码位置**：`README.md` 第 94 行

三个灵感来源各有侧重：

- **Google Pregel**：提供了 BSP 计算模型和执行语义
- **Apache Beam**：提供了 channel（类似 PCollection）和数据流水线的思想
- **NetworkX**：提供了图构建的 API 风格（`add_node`、`add_edge`）

Pregel 论文提出了 **Bulk Synchronous Parallel（BSP，整体同步并行）** 计算模型：

1. **超级步（Superstep）**：计算被组织成一系列离散的步骤
2. **消息传递**：顶点之间通过消息传递通信
3. **同步屏障（Barrier）**：每个超级步结束后，所有消息同时可见
4. **投票停止（Vote to Halt）**：当没有顶点需要执行时，计算终止

### 1.2.2 BSP 模型在 LangGraph 中的实现

LangGraph 在其核心运行时 `Pregel` 类中忠实地实现了 BSP 模型。打开
`libs/langgraph/langgraph/pregel/main.py`，可以看到 `Pregel` 类的文档字符串
精确描述了 BSP 模型在 LangGraph 中的映射：

```python
# 文件: libs/langgraph/langgraph/pregel/main.py

class Pregel(
    PregelProtocol[StateT, ContextT, InputT, OutputT],
    Generic[StateT, ContextT, InputT, OutputT],
):
    """Pregel manages the runtime behavior for LangGraph applications.

    ## Overview

    Pregel combines **actors** and **channels** into a single application.
    **Actors** read data from channels and write data to channels.
    Pregel organizes the execution of the application into multiple steps,
    following the **Pregel Algorithm**/**Bulk Synchronous Parallel** model.

    Each step consists of three phases:

    - **Plan**: Determine which **actors** to execute in this step.
        For example, in the first step, select the **actors** that
        subscribe to the special **input** channels; in subsequent steps,
        select the **actors** that subscribe to channels updated in the
        previous step.
    - **Execution**: Execute all selected **actors** in parallel,
        until all complete, or one fails, or a timeout is reached.
        During this phase, channel updates are invisible to actors
        until the next step.
    - **Update**: Update the channels with the values written by the
        **actors** in this step.

    Repeat until no **actors** are selected for execution, or a maximum
    number of steps is reached.
    """
```

> **源码位置**：`libs/langgraph/langgraph/pregel/main.py` 第 332-360 行

这段 docstring 揭示了 LangGraph 的**三阶段执行循环**：

1. **Plan（规划）**：决定本轮要执行哪些 actor（节点）。第一步执行订阅了输入
   channel 的 actor；后续步骤执行订阅了上一步被更新的 channel 的 actor。

2. **Execution（执行）**：并行执行所有被选中的 actor。关键点在于——**执行期间
   channel 的更新对其他 actor 不可见**，这保证了确定性。

3. **Update（更新）**：将本轮 actor 写入的数据同步到 channel 中。

这个 Plan-Execute-Update 循环不断重复，直到没有 actor 需要执行，或达到最大步数
限制（`recursion_limit`）。超过步数限制时，框架抛出 `GraphRecursionError`：

```python
# 文件: libs/langgraph/langgraph/errors.py

class GraphRecursionError(RecursionError):
    """Raised when the graph has exhausted the maximum number of steps.

    This prevents infinite loops. To increase the maximum number of steps,
    run your graph with a config specifying a higher `recursion_limit`.

    Examples:

        graph = builder.compile()
        graph.invoke(
            {"messages": [("user", "Hello, world!")]},
            # The config is the second positional argument
            {"recursion_limit": 1000},
        )
    """
```

> **源码位置**：`libs/langgraph/langgraph/errors.py` 第 45-63 行


### 1.2.3 BSP 模型到 LangGraph 的概念映射

下面这张表展示了 Pregel 论文中的概念如何映射到 LangGraph 的实现：

| Pregel 论文概念 | LangGraph 映射 | 源码位置 |
|----------------|---------------|---------|
| Vertex（顶点） | `PregelNode`（Actor） | `libs/langgraph/langgraph/pregel/_read.py` |
| Edge（边） | Channel（通道） | `libs/langgraph/langgraph/channels/` |
| Message（消息） | Channel Update（通道更新） | `libs/langgraph/langgraph/channels/base.py` |
| Superstep（超级步） | Step（步骤） | `libs/langgraph/langgraph/pregel/_loop.py` |
| Vote to Halt | 没有 Actor 被选中执行 | `libs/langgraph/langgraph/pregel/_algo.py` |
| Combiner | Reducer（归约函数） | `libs/langgraph/langgraph/channels/binop.py` |
| Aggregator | `BinaryOperatorAggregate` | `libs/langgraph/langgraph/channels/binop.py` |

这个映射有几个关键的设计决策值得注意。

**决策 1：通道（Channel）替代显式消息传递**

在原始 Pregel 中，顶点之间通过消息传递通信。LangGraph 引入了**通道（Channel）**
抽象——通道既是通信媒介，又是状态存储。这意味着 LangGraph 中的"消息"不是临时的，
而是持久化在通道中的。这使得 checkpoint 和恢复执行变得自然。

通道的基类定义在 `libs/langgraph/langgraph/channels/base.py`：

```python
# 文件: libs/langgraph/langgraph/channels/base.py

class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    """Base class for all channels."""

    __slots__ = ("key", "typ")

    def __init__(self, typ: Any, key: str = "") -> None:
        self.typ = typ
        self.key = key

    @property
    @abstractmethod
    def ValueType(self) -> Any:
        """The type of the value stored in the channel."""

    @property
    @abstractmethod
    def UpdateType(self) -> Any:
        """The type of the update received by the channel."""
```

> **源码位置**：`libs/langgraph/langgraph/channels/base.py` 第 19-37 行

每个 Channel 有三个关键属性：
- **Value 类型**：通道中存储的值的类型
- **Update 类型**：通道接收的更新的类型
- **Checkpoint 类型**：通道序列化后的快照类型

LangGraph 提供了多种内置通道类型，每种对应不同的状态管理语义：

| 通道类型 | 文件 | 用途 | 行为 |
|---------|------|------|------|
| `LastValue` | `channels/last_value.py` | State 字段的默认通道 | 保存最后写入的值 |
| `BinaryOperatorAggregate` | `channels/binop.py` | 带 reducer 的字段 | 通过二元运算符聚合 |
| `Topic` | `channels/topic.py` | 发布-订阅 | 多值传递和累积 |
| `EphemeralValue` | `channels/ephemeral_value.py` | 临时数据 | 每步重置 |
| `AnyValue` | `channels/any_value.py` | 任意值 | 接受任意类型 |
| `NamedBarrierValue` | `channels/named_barrier_value.py` | 同步屏障 | 等待所有来源写入 |
| `UntrackedValue` | `channels/untracked_value.py` | 无版本跟踪 | 不触发订阅者 |

在 `StateGraph` 的上下文中，每个 State 字段自动映射为一个 Channel：
- 没有 reducer 注解的字段 -> `LastValue` 通道
- 使用 `Annotated[type, reducer]` 注解的字段 -> `BinaryOperatorAggregate` 通道

**决策 2：Actor 模型与 Runnable 接口的结合**

LangGraph 中的 Actor 实现为 `PregelNode`，它同时实现了 LangChain 的 `Runnable`
接口。这意味着每个节点既是 BSP 模型中的计算单元，又可以与 LangChain 生态系统
无缝集成。`Pregel` 类本身也继承了 `PregelProtocol`，使编译后的图可以直接调用
`invoke()`、`stream()`、`ainvoke()` 等标准方法。

**决策 3：确定性执行保证**

BSP 模型中的同步屏障保证了一个重要性质：**同一超级步内的节点互相看不到对方的
写入**。这意味着给定相同的初始状态和输入，无论节点的实际执行顺序如何，最终结果
都是确定的。这对于调试和测试至关重要。


### 1.2.4 为什么选择图计算模型

在 Agent 编排框架的设计空间中，有多种可能的架构选择：

| 架构 | 代表 | 优势 | 劣势 |
|------|------|------|------|
| 链式（Chain） | LangChain LCEL | 简单直观 | 无法表达循环 |
| DAG | Airflow, Prefect | 成熟的调度生态 | 不支持循环和动态路由 |
| 状态机 | 传统 FSM | 形式化验证容易 | 状态爆炸问题 |
| **图计算（BSP）** | **LangGraph** | **支持循环、并行、持久化** | **概念较多，学习曲线较陡** |
| Actor 系统 | Akka, Ray | 完全异步 | 缺乏同步屏障和确定性 |

LangGraph 选择 BSP 图计算模型的关键原因：

1. **支持循环**：Agent 需要反复思考、多轮工具调用。图天然支持循环，而 DAG 不行。
2. **并行性**：BSP 模型中同一超级步内的节点天然并行。
3. **确定性**：同步屏障保证了执行的确定性——给定相同的初始状态和输入，执行结果
   是可预测的。
4. **可检查点**：超级步之间的状态是完整的、一致的，天然适合持久化。
5. **可视化**：图结构天然可视化，便于理解和调试。

这种映射使得 LangGraph 天然支持**循环（cycle）**——这正是 Agent 场景的刚需
（思考-行动-观察循环），而传统的 DAG 编排框架无法优雅表达。


## 1.3 核心价值：四大支柱

`README.md` 列出了 LangGraph 的核心价值。让我们逐一解析，并追溯到源码中的
实现基础。

### 1.3.1 持久执行（Durable Execution）

> Build agents that persist through failures and can run for extended periods,
> automatically resuming from exactly where they left off.

持久执行是 LangGraph 区别于简单 Agent 框架的核心能力。它指的是：Agent 的执行
可以在任意步骤被中断（机器故障、进程重启、人工干预），并在之后从中断点精确恢复。

这个能力依赖两个基础设施。

**1. Checkpoint 数据结构**

每个超级步结束后，LangGraph 将完整的状态快照保存为 checkpoint。checkpoint 的
核心数据结构定义在 `libs/checkpoint/langgraph/checkpoint/base/__init__.py`：

```python
# 文件: libs/checkpoint/langgraph/checkpoint/base/__init__.py

class Checkpoint(TypedDict):
    """State snapshot at a given point in time."""

    v: int
    """The version of the checkpoint format. Currently `1`."""
    id: str
    """The ID of the checkpoint.

    This is both unique and monotonically increasing, so can be used for
    sorting checkpoints from first to last."""
    ts: str
    """The timestamp of the checkpoint in ISO 8601 format."""
    channel_values: dict[str, Any]
    """The values of the channels at the time of the checkpoint.

    Mapping from channel name to deserialized channel snapshot value."""
```

> **源码位置**：`libs/checkpoint/langgraph/checkpoint/base/__init__.py` 第 65-80 行

注意 `id` 字段的设计——它既是唯一的，又是单调递增的（使用 UUID v6，定义在
`langgraph.checkpoint.base.id` 模块中）。这使得 checkpoint 可以按时间顺序排列，
支持"时间旅行"调试——你可以回到任何一个历史 checkpoint，从那里重新运行。

**2. CheckpointMetadata**

每个 checkpoint 附带元数据，记录 checkpoint 的来源和上下文：

```python
# 文件: libs/checkpoint/langgraph/checkpoint/base/__init__.py

class CheckpointMetadata(TypedDict, total=False):
    """Metadata associated with a checkpoint."""

    source: Literal["input", "loop", "update", "fork"]
    """The source of the checkpoint.

    - "input": The checkpoint was created from an input to invoke/stream/batch.
    - "loop": The checkpoint was created from inside the pregel loop.
    - "update": The checkpoint was created from a manual state update.
    - "fork": The checkpoint was created as a copy of another checkpoint.
    """
    step: int
    """The step number of the checkpoint.

    -1 for the first "input" checkpoint.
    0 for the first "loop" checkpoint.
    """
    parents: dict[str, str]
    """The IDs of the parent checkpoints.

    Mapping from checkpoint namespace to checkpoint ID.
    """
    run_id: str
    """The ID of the run that created this checkpoint."""
```

> **源码位置**：`libs/checkpoint/langgraph/checkpoint/base/__init__.py` 第 35-60 行

`source` 字段的四种取值清楚地描述了 checkpoint 的生命周期：

- `"input"`：用户输入触发的初始 checkpoint（step = -1）
- `"loop"`：Pregel 循环内部创建的 checkpoint（step = 0, 1, 2, ...）
- `"update"`：手动状态更新创建的 checkpoint
- `"fork"`：从其他 checkpoint 复制的 checkpoint（用于分支探索）

**3. Checkpoint 存储后端**

LangGraph 将 checkpoint 存储抽象为接口（`BaseCheckpointSaver`），提供了多种实现：

| 后端 | 包名 | 版本 | 适用场景 |
|------|------|------|---------|
| 内存 | `langgraph-checkpoint` (InMemorySaver) | 4.0.1 | 开发、测试 |
| SQLite | `langgraph-checkpoint-sqlite` | 3.0.3 | 单机轻量部署 |
| PostgreSQL | `langgraph-checkpoint-postgres` | 3.0.4 | 生产环境 |

这种分层设计使得开发者可以在开发阶段使用内存存储快速迭代，在生产环境切换到
PostgreSQL 而无需修改任何业务代码——只需更换 checkpointer 实例。

**4. 序列化机制**

Checkpoint 的序列化使用 `ormsgpack`（MessagePack 格式）作为默认序列化器，并提供
了 `JsonPlusSerializer` 和 `EncryptedSerializer` 等替代方案。序列化相关代码位于
`libs/checkpoint/langgraph/checkpoint/serde/` 目录。


### 1.3.2 Human-in-the-Loop（人机协作）

> Seamlessly incorporate human oversight by inspecting and modifying agent state
> at any point during execution.

Human-in-the-loop 建立在持久执行之上。LangGraph 通过 **interrupt** 机制实现
这一能力。

`interrupt` 函数允许节点在执行过程中暂停，等待人类输入。interrupt 机制的工作
原理：

1. 节点调用 `interrupt(value)` 抛出特殊异常（`GraphBubbleUp` 的子类）
2. Pregel 运行时捕获异常，将当前状态保存为 checkpoint
3. `value` 作为 interrupt 数据返回给调用方
4. 人类审查后，通过 `Command(resume=...)` 恢复执行
5. 恢复时，`interrupt()` 返回人类提供的值

这个机制的优雅之处在于：对于节点函数来说，`interrupt()` 看起来就像一个普通的
函数调用——它暂停、等待、然后返回一个值。所有的持久化和恢复逻辑对业务代码完全
透明。

`Command` 类型定义在 `libs/langgraph/langgraph/types.py`，它是 LangGraph
控制流的核心原语之一。`Command` 不仅用于恢复 interrupt，还可以用于：

- **路由到指定节点**：`Command(goto="node_name")`
- **更新图状态**：`Command(update={...})`
- **恢复中断**：`Command(resume=value)`

这使得 Human-in-the-loop 不仅仅是简单的"批准/拒绝"，而是可以修改状态、
改变执行路径的完整交互模型。

LangGraph 还支持在编译时声明中断点：

```python
compiled = graph.compile(
    checkpointer=memory,
    interrupt_before=["human_review_node"],
    interrupt_after=["tool_call_node"],
)
```

`interrupt_before` 在指定节点执行前暂停，`interrupt_after` 在执行后暂停。
这提供了更声明式的 Human-in-the-loop 配置方式。


### 1.3.3 全面记忆（Comprehensive Memory）

> Create truly stateful agents with both short-term working memory for ongoing
> reasoning and long-term persistent memory across sessions.

LangGraph 的记忆系统分为两个层次。

**短期记忆（Working Memory）**

通过图状态（State）实现。在单次运行中，状态在节点之间流转，每个节点可以读取和
更新状态。这是图计算模型自带的能力。

在 `libs/langgraph/langgraph/graph/state.py` 中，`StateGraph` 类展示了状态
定义的核心模式：

```python
# 文件: libs/langgraph/langgraph/graph/state.py

class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    """A graph whose nodes communicate by reading and writing to a shared state.

    The signature of each node is State -> Partial<State>.

    Each state key can optionally be annotated with a reducer function that
    will be used to aggregate the values of that key received from multiple
    nodes. The signature of a reducer function is (Value, Value) -> Value.
    """
```

> **源码位置**：`libs/langgraph/langgraph/graph/state.py` 第 115-123 行

Reducer 机制是理解 LangGraph 记忆系统的关键。当使用 `Annotated[type, reducer]`
标注状态字段时，多个节点对同一字段的更新不会覆盖，而是通过 reducer 函数聚合：

```python
from typing_extensions import Annotated, TypedDict


def reducer(a: list, b: int | None) -> list:
    if b is not None:
        return a + [b]
    return a


class State(TypedDict):
    x: Annotated[list, reducer]
```

> **源码位置**：`libs/langgraph/langgraph/graph/state.py` 第 152-159 行（文档示例）

在通道系统层面，reducer 对应的是 `BinaryOperatorAggregate` 通道
（`libs/langgraph/langgraph/channels/binop.py`）。当状态字段没有 reducer 时，
默认使用 `LastValue` 通道——最后一次写入的值覆盖之前的值。

**长期记忆（Persistent Memory）**

通过 checkpoint 机制实现。跨会话的状态可以持久化到数据库，下次运行时恢复。结合
`thread_id` 配置，可以实现多会话的独立记忆。此外，`BaseStore` 接口提供了更通用
的键值存储，可以在多次对话之间保持用户偏好、学习到的知识等。

**消息列表：最常见的记忆模式**

在聊天 Agent 场景中，最常见的短期记忆模式是消息列表。LangGraph 提供了内置的
`MessagesState` 和 `add_messages` reducer：

```python
# 文件: libs/langgraph/langgraph/graph/__init__.py

from langgraph.graph.message import MessageGraph, MessagesState, add_messages
```

> **源码位置**：`libs/langgraph/langgraph/graph/__init__.py` 第 2 行

`add_messages` 是一个预定义的 reducer，负责将新消息追加到消息列表中，同时处理
消息 ID 去重等边界情况。`MessagesState` 则是一个预定义的 State schema：

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

这使得构建聊天 Agent 只需：`StateGraph(MessagesState)`，而不需要手动定义
消息列表和 reducer。


### 1.3.4 生产就绪部署（Production-Ready Deployment）

> Deploy sophisticated agent systems confidently with scalable infrastructure
> designed to handle the unique challenges of stateful, long-running workflows.

LangGraph 的生产部署能力体现在以下几个方面。

**1. CLI 工具（`langgraph-cli`）**

`langgraph-cli` 提供了部署相关的命令行工具。其入口点定义在
`libs/cli/pyproject.toml`：

```toml
# 文件: libs/cli/pyproject.toml

[project.scripts]
langgraph = "langgraph_cli.cli:cli"
```

> **源码位置**：`libs/cli/pyproject.toml` 第 35 行

CLI 模块（`libs/cli/langgraph_cli/`）包含丰富的功能：

```
libs/cli/langgraph_cli/
├── cli.py            # 命令行入口（基于 click）
├── config.py         # 配置管理
├── docker.py         # Docker 化部署
├── exec.py           # 执行管理
├── host_backend.py   # 主机后端
├── progress.py       # 进度显示
├── schemas.py        # 配置 schema
├── templates.py      # 项目模板
├── analytics.py      # 分析追踪
└── constants.py      # CLI 常量
```

**2. SDK（`langgraph-sdk`）**

`langgraph-sdk` 是与 LangGraph 部署服务交互的客户端 SDK，使用 HTTP 协议通信。
它不依赖 LangGraph 核心库，可以独立使用：

```toml
# 文件: libs/sdk-py/pyproject.toml

dependencies = ["httpx>=0.25.2", "orjson>=3.11.5"]
```

> **源码位置**：`libs/sdk-py/pyproject.toml` 第 14 行

SDK 的模块结构包括同步和异步客户端、错误处理、SSE 流式支持等：

```
libs/sdk-py/langgraph_sdk/
├── _async/           # 异步客户端
├── _shared/          # 共享工具
├── _sync/            # 同步客户端
├── auth/             # 认证
├── client.py         # 客户端入口
├── encryption/       # 加密
├── errors.py         # 错误定义
├── runtime.py        # 运行时
├── schema.py         # 数据模型
└── sse.py            # Server-Sent Events
```

**3. 错误处理与重试**

生产环境中的 Agent 需要健壮的错误处理。LangGraph 提供了完善的错误类型体系，
定义在 `libs/langgraph/langgraph/errors.py`：

```python
# 文件: libs/langgraph/langgraph/errors.py

class ErrorCode(Enum):
    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"
```

> **源码位置**：`libs/langgraph/langgraph/errors.py` 第 29-34 行

每种错误码都有对应的文档链接，通过 `create_error_message` 函数生成包含
troubleshooting URL 的错误消息：

```python
# 文件: libs/langgraph/langgraph/errors.py

def create_error_message(*, message: str, error_code: ErrorCode) -> str:
    return (
        f"{message}\n"
        "For troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/"
        f"errors/{error_code.value}"
    )
```

> **源码位置**：`libs/langgraph/langgraph/errors.py` 第 37-42 行

此外，`RetryPolicy` 和 `CachePolicy`（定义在 `libs/langgraph/langgraph/types.py`）
提供了节点级别的重试和缓存策略，使得每个节点可以根据业务需求配置不同的容错行为。

**4. LangSmith 集成**

通过 LangChain 的 callback 系统，LangGraph 与 LangSmith 深度集成，提供：

- 执行路径追踪
- 状态变迁可视化
- 运行时指标
- Agent 评估

`TAG_HIDDEN` 常量（定义在 `libs/langgraph/langgraph/constants.py`）用于标记
不应出现在 tracing 中的内部节点：

```python
# 文件: libs/langgraph/langgraph/constants.py

TAG_HIDDEN = sys.intern("langsmith:hidden")
"""Tag to hide a node/edge from certain tracing/streaming environments."""
```

> **源码位置**：`libs/langgraph/langgraph/constants.py` 第 26 行


## 1.4 两种 API：Graph API 与 Functional API

LangGraph 提供了两种等价的编程接口，它们最终都编译为 `Pregel` 运行时。

### 1.4.1 Graph API（StateGraph）

Graph API 是最常用的接口，通过显式的图构建器模式定义工作流：

```python
from langgraph.graph import START, StateGraph

graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")
compiled = graph.compile()
```

`StateGraph` 类定义在 `libs/langgraph/langgraph/graph/state.py`，它是一个
**builder（构建器）**，不能直接执行。必须调用 `.compile()` 方法将其编译为
可执行的 `CompiledStateGraph`（继承自 `Pregel`），后者才是真正的运行时引擎。

`StateGraph` 的初始化签名展示了它的核心概念：

```python
# 文件: libs/langgraph/langgraph/graph/state.py

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

> **源码位置**：`libs/langgraph/langgraph/graph/state.py` 第 200-208 行

四个 schema 参数定义了图的类型边界：
- `state_schema`：**必填**，定义图内部的完整状态
- `context_schema`：可选，定义运行时上下文（如 `user_id`、`db_conn` 等不可变数据）
- `input_schema`：可选，定义图的外部输入（默认等同于 state_schema）
- `output_schema`：可选，定义图的外部输出（默认等同于 state_schema）

StateGraph 的核心属性反映了图编译前需要收集的所有信息：

```python
# 文件: libs/langgraph/langgraph/graph/state.py

class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
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

> **源码位置**：`libs/langgraph/langgraph/graph/state.py` 第 186-198 行

`StateGraph` 的核心方法：

| 方法 | 作用 | 说明 |
|------|------|------|
| `add_node(name, action)` | 添加节点 | 支持自动推断名称 |
| `add_edge(start, end)` | 添加普通边 | 定义固定路由 |
| `add_conditional_edges(source, path)` | 添加条件路由边 | 动态路由 |
| `set_entry_point(key)` | 设置入口节点 | 等价于 `add_edge(START, key)` |
| `set_finish_point(key)` | 设置终止节点 | 等价于 `add_edge(key, END)` |
| `compile(...)` | 编译为可执行图 | 返回 `CompiledStateGraph` |


### 1.4.2 Functional API（entrypoint / task）

Functional API 是后来引入的声明式接口，位于 `libs/langgraph/langgraph/func/`
目录。它通过 `@entrypoint` 和 `@task` 装饰器定义工作流，更接近普通 Python 函数
的写法：

```python
from langgraph.func import entrypoint, task

@task
def step_a(text: str) -> str:
    return text + "a"

@task
def step_b(text: str) -> str:
    return text + "b"

@entrypoint()
def my_workflow(text: str) -> str:
    result = step_a(text).result()
    return step_b(result).result()
```

Functional API 的优势在于代码更加线性、可读性更好，尤其适合简单的顺序工作流。
但在底层，它同样编译为 `Pregel` 运行时，享受相同的持久化和 Human-in-the-loop
能力。

### 1.4.3 底层统一：Pregel

无论使用哪种 API，最终都编译为 `Pregel` 对象。这个统一性是 LangGraph 架构的
核心优势——所有上层 API 共享同一个运行时，享受相同的执行语义、持久化能力和
调试工具。

`Pregel` 也可以直接使用（低层次 API），用于更精细的控制。它使用 `NodeBuilder`
模式构建节点：

```python
# 文件: libs/langgraph/langgraph/pregel/main.py（文档示例）

from langgraph.channels import EphemeralValue
from langgraph.pregel import Pregel, NodeBuilder

node1 = (
    NodeBuilder().subscribe_only("a")
    .do(lambda x: x + x)
    .write_to("b")
)
```

> **源码位置**：`libs/langgraph/langgraph/pregel/main.py` 第 414-418 行

`NodeBuilder` 提供了链式构建接口：

| 方法 | 作用 |
|------|------|
| `subscribe_only(channel)` | 订阅单个通道作为输入 |
| `subscribe_to(*channels)` | 订阅多个通道 |
| `read_from(*channels)` | 额外读取通道（不触发执行） |
| `do(node)` | 设置节点执行函数 |
| `write_to(*channels)` | 设置输出通道 |
| `meta(*tags, **metadata)` | 添加标签和元数据 |
| `add_retry_policies(*policies)` | 添加重试策略 |
| `add_cache_policy(policy)` | 添加缓存策略 |
| `build()` | 构建 `PregelNode` |

> **源码位置**：`libs/langgraph/langgraph/pregel/main.py` 第 168-329 行

Builder 模式清楚地展示了 `PregelNode` 的本质：一个订阅了某些通道、执行某个函数、
并将结果写入其他通道的计算单元。这正是 BSP 模型中"顶点"的具体实现。

### 1.4.4 Builder -> Compiler -> Runtime 架构总结

```
用户层 API                        底层运行时
==========                       ===========

StateGraph（Graph API）
        │
        │ .compile()
        ▼
CompiledStateGraph ─────────── 继承自 ─── Pregel
        ▲                                   │
        │ .compile()                        │ 使用
        │                                   ▼
@entrypoint（Functional API）         Channels + PregelNodes
                                    （通信与计算原语）
```

这个分层设计保证了：
- **定义与执行分离**：图定义是静态的、可序列化的；运行时是动态的、有状态的
- **API 可替换**：可以用不同的 API 定义同一个图
- **运行时统一**：所有 API 共享同一个经过充分测试的 Pregel 运行时


## 1.5 与 LangChain 的关系：互补而非替代

### 1.5.1 依赖关系分析

从 `libs/langgraph/pyproject.toml` 可以看到 LangGraph 核心包的依赖：

```toml
# 文件: libs/langgraph/pyproject.toml

dependencies = [
    "langchain-core>=0.1",
    "langgraph-checkpoint>=2.1.0,<5.0.0",
    "langgraph-sdk>=0.3.0,<0.4.0",
    "langgraph-prebuilt>=1.0.8,<1.1.0",
    "xxhash>=3.5.0",
    "pydantic>=2.7.4",
]
```

> **源码位置**：`libs/langgraph/pyproject.toml` 第 27-33 行

关键观察：

**1. 依赖 `langchain-core`，而非 `langchain`**

LangGraph 只依赖 `langchain-core`，这是 LangChain 的最小核心包，主要提供
`Runnable` 接口、`RunnableConfig` 类型和回调基础设施。LangGraph 不依赖
LangChain 的模型集成（`ChatOpenAI` 等）、链抽象（`LLMChain` 等）或工具框架。

**2. `langgraph-checkpoint` 也依赖 `langchain-core`**

```toml
# 文件: libs/checkpoint/pyproject.toml

dependencies = [
    "langchain-core>=0.2.38",
    "ormsgpack>=1.12.0",
]
```

> **源码位置**：`libs/checkpoint/pyproject.toml` 第 14-17 行

checkpoint 包需要 `langchain-core` 主要是因为 `RunnableConfig` 类型和序列化
支持。

**3. `langgraph-sdk` 完全独立**

SDK 包不依赖任何 LangChain 组件：

```toml
# 文件: libs/sdk-py/pyproject.toml

dependencies = ["httpx>=0.25.2", "orjson>=3.11.5"]
```

> **源码位置**：`libs/sdk-py/pyproject.toml` 第 14 行

这意味着你可以在一个不安装 LangGraph 核心库的环境中使用 SDK 与远程部署的
LangGraph 服务交互。

**4. `langgraph-prebuilt` 依赖 `langchain-core` 但不依赖完整 `langgraph`**

```toml
# 文件: libs/prebuilt/pyproject.toml

dependencies = [
    "langgraph-checkpoint>=2.1.0,<5.0.0",
    "langchain-core>=1.0.0",
]
```

> **源码位置**：`libs/prebuilt/pyproject.toml` 第 27-29 行

prebuilt 包提供的是高层 Agent 组件（如 `create_react_agent`），它依赖
`langchain-core` 的模型接口。


### 1.5.2 接口层面的集成

LangGraph 与 LangChain 的集成主要在接口层面：

**1. Runnable 接口**

`Pregel` 通过 `PregelProtocol` 实现了 LangChain 的 `Runnable` 接口。这意味着
编译后的图可以作为 LangChain chain 的一部分使用，也可以被 LangChain 的工具
调用。

**2. RunnableConfig**

LangGraph 使用 LangChain 的 `RunnableConfig` 传递配置信息（如回调、标签、
`thread_id`、`checkpoint_id` 等）。在源码中可以看到大量的 config 处理代码：

```python
# 文件: libs/langgraph/langgraph/graph/state.py

from langchain_core.runnables import Runnable, RunnableConfig
```

> **源码位置**：`libs/langgraph/langgraph/graph/state.py` 第 27 行

**3. Callback 系统**

LangGraph 利用 LangChain 的 callback 系统实现 tracing 和可观测性：

```python
# 文件: libs/langgraph/langgraph/pregel/main.py

from langchain_core.runnables.config import (
    RunnableConfig,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
)
```

> **源码位置**：`libs/langgraph/langgraph/pregel/main.py` 第 38-41 行


### 1.5.3 互补关系总结

| 维度 | LangChain | LangGraph |
|------|-----------|-----------|
| 定位 | 模型集成与组合 | Agent 编排与运行时 |
| 抽象层级 | 高层（链、工具、记忆） | 低层（图、通道、检查点） |
| 核心依赖 | 各种模型 SDK | `langchain-core`（Runnable 接口） |
| 核心价值 | 模型标准化、工具生态 | 状态管理、持久执行 |
| 是否必须 | 否（LangGraph 只需 langchain-core） | 否（可独立使用） |
| 组合使用 | LangChain 模型 + LangGraph 编排 | 最佳实践 |

如 README 所述：

> LangGraph is built by LangChain Inc, the creators of LangChain, but can be
> used without LangChain.

简而言之：**LangChain 提供组件，LangGraph 编排组件**。你可以单独使用 LangGraph
（搭配任何 LLM 库），也可以将两者结合获得最佳开发体验。


## 1.6 核心常量与公共 API

### 1.6.1 公共常量

LangGraph 的公共常量定义在 `libs/langgraph/langgraph/constants.py`：

```python
# 文件: libs/langgraph/langgraph/constants.py

import sys

TAG_NOSTREAM = sys.intern("nostream")
"""Tag to disable streaming for a chat model."""

TAG_HIDDEN = sys.intern("langsmith:hidden")
"""Tag to hide a node/edge from certain tracing/streaming environments."""

END = sys.intern("__end__")
"""The last (maybe virtual) node in graph-style Pregel."""

START = sys.intern("__start__")
"""The first (maybe virtual) node in graph-style Pregel."""
```

> **源码位置**：`libs/langgraph/langgraph/constants.py` 第 24-31 行

注意 `sys.intern()` 的使用——这是一个 Python 优化技巧，确保这些常用字符串在内存
中只存在一份拷贝，并且可以使用 `is` 而非 `==` 进行比较，提升性能。

`START` 和 `END` 是图中的**虚拟节点**：
- `START` (`"__start__"`)：图的入口，连接到第一个实际执行的节点
- `END` (`"__end__"`)：图的出口，标记执行结束

此外，`constants.py` 还实现了向后兼容的延迟导入机制。当用户试图从 `constants`
模块导入已弃用的常量（如 `Send`、`Interrupt`）时，会得到弃用警告并自动重定向
到正确的模块：

```python
# 文件: libs/langgraph/langgraph/constants.py

def __getattr__(name: str) -> Any:
    if name in ["Send", "Interrupt"]:
        warn(
            f"Importing {name} from langgraph.constants is deprecated. "
            f"Please use 'from langgraph.types import {name}' instead.",
            LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        from importlib import import_module
        module = import_module("langgraph.types")
        return getattr(module, name)
```

> **源码位置**：`libs/langgraph/langgraph/constants.py` 第 34-46 行

这种模块级 `__getattr__` 模式是 Python 3.7+ 引入的特性，LangGraph 使用它实现了
优雅的 API 迁移——旧代码仍然能工作，但会收到弃用警告。


### 1.6.2 公共 API 导出

`libs/langgraph/langgraph/graph/__init__.py` 定义了图模块的公共 API：

```python
# 文件: libs/langgraph/langgraph/graph/__init__.py

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

> **源码位置**：`libs/langgraph/langgraph/graph/__init__.py`

最常用的导入是：

```python
from langgraph.graph import START, END, StateGraph, MessagesState
```

### 1.6.3 类型系统

LangGraph 的类型定义在 `libs/langgraph/langgraph/types.py`，这是一个大文件，
包含了框架中几乎所有的公共类型。核心导出包括：

```python
# 文件: libs/langgraph/langgraph/types.py

__all__ = (
    "All",
    "Checkpointer",
    "StreamMode",
    "StreamWriter",
    "StreamPart",
    "RetryPolicy",
    "CachePolicy",
    "Interrupt",
    "StateUpdate",
    "PregelTask",
    "PregelExecutableTask",
    "StateSnapshot",
    "Send",
    "Command",
    "Durability",
    "interrupt",
    "Overwrite",
    ...
)
```

> **源码位置**：`libs/langgraph/langgraph/types.py` 第 51-80 行

几个最重要的类型：

| 类型 | 用途 | 使用场景 |
|------|------|---------|
| `Send` | 动态创建节点调用 | fan-out 模式（一对多） |
| `Command` | 控制图执行 | 路由、更新状态、恢复 interrupt |
| `Interrupt` | 中断数据 | Human-in-the-loop 的数据载体 |
| `RetryPolicy` | 节点重试策略 | 外部 API 调用容错 |
| `CachePolicy` | 节点缓存策略 | 减少重复计算 |
| `StateSnapshot` | 图状态快照 | 调试和状态检查 |
| `StreamMode` | 流式输出模式 | `"values"`, `"updates"`, `"messages"` 等 |
| `Checkpointer` | Checkpointer 类型别名 | 类型标注 |
| `Durability` | 持久化级别 | 控制 checkpoint 频率 |


## 1.7 源码组织概览

在深入后续章节之前，先从宏观视角了解源码的组织方式。LangGraph 的核心包
`libs/langgraph/langgraph/` 的目录结构如下：

```
libs/langgraph/langgraph/
├── _internal/          # 内部实现（不属于公共 API，以下划线开头）
│   ├── _config.py      #   配置处理
│   ├── _constants.py   #   内部常量（CONF, TASKS, CONFIG_KEY_* 等）
│   ├── _fields.py      #   字段处理（从 TypedDict 提取字段信息）
│   ├── _pydantic.py    #   Pydantic 模型动态创建
│   ├── _runnable.py    #   Runnable 工具（coerce_to_runnable）
│   ├── _serde.py       #   序列化/反序列化
│   ├── _typing.py      #   类型工具（MISSING 哨兵值）
│   ├── _cache.py       #   缓存键计算
│   ├── _retry.py       #   重试逻辑
│   ├── _scratchpad.py  #   Pregel 临时存储
│   └── _queue.py       #   异步/同步队列
├── channels/           # 通道实现（节点间通信）
│   ├── base.py         #   BaseChannel 基类
│   ├── last_value.py   #   LastValue 通道（默认）
│   ├── binop.py        #   BinaryOperatorAggregate 通道（reducer）
│   ├── topic.py        #   Topic 通道（发布-订阅）
│   ├── ephemeral_value.py  # EphemeralValue 通道（每步重置）
│   ├── any_value.py    #   AnyValue 通道
│   ├── named_barrier_value.py  # NamedBarrierValue 通道（同步屏障）
│   └── untracked_value.py  # UntrackedValue 通道（不跟踪版本）
├── graph/              # 图定义 API（用户界面层）
│   ├── __init__.py     #   公共 API 导出
│   ├── state.py        #   StateGraph / CompiledStateGraph（1752 行）
│   ├── message.py      #   MessagesState / MessageGraph / add_messages
│   ├── _branch.py      #   条件分支规格
│   ├── _node.py        #   节点规格（StateNodeSpec）
│   └── ui.py           #   UI 相关
├── pregel/             # Pregel 运行时（核心执行引擎）
│   ├── __init__.py     #   导出 Pregel, NodeBuilder
│   ├── main.py         #   Pregel 类（3667 行）— 最大的单文件
│   ├── _algo.py        #   核心调度算法（1233 行）
│   ├── _loop.py        #   执行循环（1404 行）
│   ├── _runner.py      #   任务运行器
│   ├── _checkpoint.py  #   检查点操作
│   ├── _read.py        #   PregelNode / ChannelRead
│   ├── _write.py       #   ChannelWrite / ChannelWriteEntry
│   ├── _io.py          #   输入输出处理
│   ├── _retry.py       #   重试逻辑
│   ├── _validate.py    #   图验证
│   ├── _draw.py        #   图可视化（Mermaid 等）
│   ├── _config.py      #   Pregel 配置
│   ├── _messages.py    #   流式消息处理
│   ├── _call.py        #   调用标识
│   ├── _utils.py       #   工具函数
│   ├── protocol.py     #   PregelProtocol（Runnable 接口实现）
│   ├── debug.py        #   调试工具
│   ├── remote.py       #   远程 Pregel（代理远程服务）
│   └── types.py        #   Pregel 类型
├── func/               # Functional API
│   └── __init__.py     #   @entrypoint / @task
├── managed/            # Managed Values
│   └── base.py         #   ManagedValue 基类
├── constants.py        # 公共常量（START, END, TAG_*）
├── types.py            # 公共类型（Command, Send, Interrupt 等）
├── errors.py           # 错误类型
├── config.py           # 配置工具
├── runtime.py          # Runtime 类
├── typing.py           # 类型变量（StateT, InputT, OutputT 等）
├── utils/              # 工具函数
├── version.py          # 版本号
└── warnings.py         # 弃用警告类
```

核心文件规模：

| 文件 | 行数 | 职责 |
|------|------|------|
| `pregel/main.py` | 3667 | Pregel 运行时完整实现 |
| `graph/state.py` | 1752 | StateGraph 构建器和编译器 |
| `pregel/_loop.py` | 1404 | 同步/异步执行循环 |
| `pregel/_algo.py` | 1233 | 任务规划和写入应用算法 |
| **合计** | **8056** | **运行时核心** |

这四个文件合计超过 **8000 行**，是 LangGraph 运行时的核心。我们将在第 3-8 章
逐一深入分析。


## 1.8 版本信息

当前分析的 LangGraph 版本信息来自各 `pyproject.toml`：

| 包名 | 版本 | Python 版本 | 构建系统 |
|------|------|-------------|---------|
| `langgraph` | 1.1.0 | >=3.10 | hatchling |
| `langgraph-checkpoint` | 4.0.1 | >=3.10 | hatchling |
| `langgraph-checkpoint-postgres` | 3.0.4 | >=3.10 | hatchling |
| `langgraph-checkpoint-sqlite` | 3.0.3 | >=3.10 | hatchling |
| `langgraph-prebuilt` | 1.0.8 | >=3.10 | hatchling |
| `langgraph-checkpoint-conformance` | 0.0.1 | >=3.10 | hatchling |
| `langgraph-cli` | 动态版本 | >=3.10 | hatchling |
| `langgraph-sdk` | 动态版本 | >=3.10 | hatchling |

版本号的获取逻辑定义在 `libs/langgraph/langgraph/version.py`：

```python
# 文件: libs/langgraph/langgraph/version.py

from importlib import metadata

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
```

> **源码位置**：`libs/langgraph/langgraph/version.py`

对于 CLI 和 SDK 等使用动态版本的包，版本从各自的 `__init__.py` 中读取（通过
`[tool.hatch.version]` 配置）。

所有 Python 包都要求 `python >= 3.10`（利用了 `match` 语句、`type | type` 联合
类型等 3.10 特性），构建系统统一使用 `hatchling`，开发依赖管理使用 `uv`。

---

## 1.9 延伸阅读

- [LangGraph 官方教程](https://langchain-ai.github.io/langgraph/tutorials/introduction/) — 从零开始构建第一个 LangGraph 应用
- [LangGraph 底层概念](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 图、节点、边、状态、通道的概念详解
- [Google Pregel 论文](https://research.google/pubs/pub37252/) — *Pregel: A System for Large-Scale Graph Processing*，LangGraph 的理论基础

---

## 1.10 本章要点

1. **定位明确**：LangGraph 是低层次 Agent 编排框架，不抽象 prompt 或架构，而是
   提供状态管理、执行调度、持久化等基础设施。

2. **Pregel 启发**：运行时核心 `Pregel` 类直接借鉴了 Google Pregel 论文的 BSP
   模型，采用 Plan-Execute-Update 三阶段循环。通道（Channel）替代了论文中的
   消息传递，同时承担通信和状态存储双重职责。

3. **四大核心价值**：
   - **持久执行**：Checkpoint 机制（UUID v6 有序 ID、四种来源类型、多种存储后端）
   - **Human-in-the-loop**：interrupt/Command 机制（暂停、审查、恢复）
   - **全面记忆**：短期 State + 长期 Checkpoint/Store
   - **生产就绪**：CLI 部署工具、独立 SDK、完善的错误处理和重试策略

4. **两种 API**：Graph API（StateGraph）和 Functional API（entrypoint/task），
   底层统一编译为 `Pregel` 运行时。还可以直接使用 `NodeBuilder` 底层 API。

5. **与 LangChain 互补**：只依赖 `langchain-core`（Runnable 接口和回调系统），
   可独立使用。LangChain 提供组件，LangGraph 编排组件。

6. **核心抽象**：
   - `StateGraph`：图构建器（Builder）
   - `CompiledStateGraph`：编译产物
   - `Pregel`：执行引擎（Runtime）
   - `BaseChannel`：通道（通信与状态存储）
   - `Checkpoint`：状态快照（持久化）
   - `Command` / `Send`：控制流原语

7. **源码规模**：核心运行时四大文件（`pregel/main.py`、`graph/state.py`、
   `pregel/_loop.py`、`pregel/_algo.py`）合计超过 8000 行代码。
