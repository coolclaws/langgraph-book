# 第 2 章 Monorepo 结构与模块依赖

LangGraph 采用 Monorepo 架构，将多个独立发布的 Python 包组织在同一个 Git 仓库中。
本章将深入分析仓库的目录结构、每个子包的职责、包之间的依赖关系，以及构建系统的
统一配置。理解这些结构性知识，是后续深入源码分析的必要前提。

---

## 2.1 Monorepo 总览

### 2.1.1 为什么选择 Monorepo

LangGraph 选择 Monorepo（单仓多包）架构而非多仓架构，有几个实际的工程原因：

1. **原子提交**：跨包的重构可以在一个 commit 中完成，避免版本不一致
2. **统一 CI/CD**：所有包共享同一套持续集成配置
3. **开发便捷**：通过 `editable install`（可编辑安装），修改一个包立即对其他
   包生效，无需反复发布
4. **一致性保证**：代码风格、测试标准、构建配置全局统一

### 2.1.2 顶层目录结构

LangGraph 仓库的 `libs/` 目录下包含 9 个子目录（8 个独立发布的包 + 1 个内部
测试套件）：

```
libs/
├── langgraph/                  # 核心框架（langgraph）
├── checkpoint/                 # Checkpoint 基础接口（langgraph-checkpoint）
├── checkpoint-postgres/        # PostgreSQL Checkpoint 实现
├── checkpoint-sqlite/          # SQLite Checkpoint 实现
├── checkpoint-conformance/     # Checkpoint 一致性测试套件（内部）
├── prebuilt/                   # 预制高层组件（langgraph-prebuilt）
├── cli/                        # 命令行工具（langgraph-cli）
├── sdk-py/                     # Python SDK（langgraph-sdk）
└── sdk-js/                     # JavaScript SDK（仅 README，实际代码在另一仓库）
```

注意 `sdk-js` 目录只包含一个 `README.md`，JavaScript SDK 的实际源码位于
独立的 [langgraphjs](https://github.com/langchain-ai/langgraphjs) 仓库。


### 2.1.3 各包版本与代码规模

| 包名 | 版本 | 源码行数 | 描述 |
|------|------|---------|------|
| `langgraph` | 1.1.0 | ~19,660 | 核心框架 |
| `langgraph-checkpoint` | 4.0.1 | ~5,470 | Checkpoint 基础接口与序列化 |
| `langgraph-checkpoint-postgres` | 3.0.4 | ~4,390 | PostgreSQL 实现 |
| `langgraph-checkpoint-sqlite` | 3.0.3 | ~3,480 | SQLite 实现 |
| `langgraph-checkpoint-conformance` | 0.0.1 | ~2,980 | 一致性测试套件 |
| `langgraph-prebuilt` | 1.0.8 | ~3,230 | 预制 Agent 组件 |
| `langgraph-cli` | 0.4.15 | ~4,780 | 命令行工具 |
| `langgraph-sdk` | 0.3.10 | ~12,380 | Python SDK |
| **合计** | | **~56,370** | |

> 行数统计基于各包 Python 源码文件（`.py`），不含测试代码。

从代码量来看，`langgraph` 核心包（约 2 万行）占据了绝对主体，其次是 `langgraph-sdk`
（约 1.2 万行）。Checkpoint 相关的三个包合计约 1.3 万行，CLI 约 4,800 行。


## 2.2 各包详细分析

### 2.2.1 `langgraph`：核心框架

> "Building stateful, multi-actor applications with LLMs"

| 属性 | 值 |
|------|------|
| 包名 | `langgraph` |
| 版本 | 1.1.0 |
| 目录 | `libs/langgraph/` |
| 发布名 | langgraph |
| Python | >=3.10 |
| License | MIT |

**pyproject.toml 依赖分析**

```toml
# 文件: libs/langgraph/pyproject.toml

[project]
name = "langgraph"
version = "1.1.0"
description = "Building stateful, multi-actor applications with LLMs"
requires-python = ">=3.10"
license = "MIT"

dependencies = [
    "langchain-core>=0.1",
    "langgraph-checkpoint>=2.1.0,<5.0.0",
    "langgraph-sdk>=0.3.0,<0.4.0",
    "langgraph-prebuilt>=1.0.8,<1.1.0",
    "xxhash>=3.5.0",
    "pydantic>=2.7.4",
]
```

> **源码位置**：`libs/langgraph/pyproject.toml` 第 5-33 行

运行时依赖逐一分析：

| 依赖 | 版本约束 | 用途 |
|------|---------|------|
| `langchain-core` | >=0.1 | Runnable 接口、RunnableConfig、回调系统 |
| `langgraph-checkpoint` | >=2.1.0,<5.0.0 | Checkpoint 基础接口和序列化 |
| `langgraph-sdk` | >=0.3.0,<0.4.0 | SDK 类型定义（被 Pregel remote 使用） |
| `langgraph-prebuilt` | >=1.0.8,<1.1.0 | 预制组件（create_react_agent 等） |
| `xxhash` | >=3.5.0 | 高性能哈希（用于 channel 版本计算和缓存键） |
| `pydantic` | >=2.7.4 | 数据验证和模型动态创建 |

版本约束的设计值得注意：

- `langgraph-checkpoint` 使用宽松的 `>=2.1.0,<5.0.0`——允许跨大版本兼容
- `langgraph-sdk` 使用严格的 `>=0.3.0,<0.4.0`——0.x 版本 API 不稳定
- `langgraph-prebuilt` 使用 `>=1.0.8,<1.1.0`——patch 级别兼容

`xxhash` 的使用值得特别说明。在 `libs/langgraph/langgraph/types.py` 中可以看到：

```python
# 文件: libs/langgraph/langgraph/types.py

from xxhash import xxh3_128_hexdigest
```

> **源码位置**：`libs/langgraph/langgraph/types.py` 第 23 行

xxh3 是 xxHash 算法族中最新、最快的变体，LangGraph 使用它来计算缓存键和通道
版本标识，性能远超 Python 内置的 `hash()` 函数。

**核心包的内部模块结构**

```
libs/langgraph/langgraph/
├── _internal/          # 内部实现（以下划线前缀标记为私有）
│   ├── _config.py      #   配置处理（ensure_config, merge_configs, patch_config 等）
│   ├── _constants.py   #   内部常量（CONF, TASKS, CONFIG_KEY_* 系列）
│   ├── _fields.py      #   字段处理（从 TypedDict 提取字段、reducer 等信息）
│   ├── _pydantic.py    #   Pydantic 动态模型创建
│   ├── _runnable.py    #   coerce_to_runnable（将函数转为 Runnable）
│   ├── _serde.py       #   序列化工具
│   ├── _typing.py      #   类型工具（MISSING 哨兵值、DeprecatedKwargs）
│   ├── _cache.py       #   缓存键计算（default_cache_key）
│   ├── _retry.py       #   重试逻辑（default_retry_on）
│   ├── _scratchpad.py  #   PregelScratchpad（步骤间临时存储）
│   └── _queue.py       #   AsyncQueue / SyncQueue
├── channels/           # 通道系统（8 种通道类型）
├── graph/              # Graph API（StateGraph, CompiledStateGraph）
├── pregel/             # Pregel 运行时（核心执行引擎，21 个文件）
├── func/               # Functional API（@entrypoint, @task）
├── managed/            # Managed Values
├── utils/              # 公共工具函数
├── constants.py        # 公共常量
├── types.py            # 公共类型
├── errors.py           # 错误定义
├── config.py           # 配置工具
├── runtime.py          # Runtime 类
├── typing.py           # 类型变量
├── version.py          # 版本号
└── warnings.py         # 弃用警告
```

`_internal/` 目录使用下划线前缀明确标记为**私有 API**——这些模块不应被外部代码
直接导入。从 `constants.py` 中可以看到对此的严格态度：

```python
# 文件: libs/langgraph/langgraph/constants.py

def __getattr__(name: str) -> Any:
    # ... 尝试从私有常量模块导入
    try:
        from importlib import import_module
        private_constants = import_module("langgraph._internal._constants")
        attr = getattr(private_constants, name)
        warn(
            f"Importing {name} from langgraph.constants is deprecated. "
            f"This constant is now private and should not be used directly. "
            "Please let the LangGraph team know if you need this value.",
            LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        return attr
    except AttributeError:
        pass
```

> **源码位置**：`libs/langgraph/langgraph/constants.py` 第 48-61 行

**开发依赖组**

```toml
# 文件: libs/langgraph/pyproject.toml

[dependency-groups]
test = [
    "pytest",
    "pytest-cov",
    "pytest-dotenv",
    "pytest-mock",
    "syrupy",
    "httpx",
    "pytest-watcher",
    "pytest-xdist[psutil]",
    "pytest-repeat",
    "langchain-core>=1.0.0",
    "langgraph-prebuilt",
    "langgraph-checkpoint",
    "langgraph-checkpoint-sqlite",
    "langgraph-checkpoint-postgres",
    "langgraph-sdk",
    "psycopg[binary]",
    "uvloop==0.22.1",
    "pyperf",
    "py-spy",
    "pycryptodome",
    "langgraph-cli; python_version < '3.14'",
    "langgraph-cli[inmem]; python_version < '3.14'",
    "redis",
]
```

> **源码位置**：`libs/langgraph/pyproject.toml` 第 46-70 行

测试依赖值得关注的几个工具：
- `syrupy`：快照测试框架（用于验证输出格式不变）
- `pytest-xdist`：并行测试执行
- `uvloop`：高性能事件循环（用于异步测试的性能基准）
- `pyperf` / `py-spy`：性能分析工具
- `pycryptodome`：加密库（用于测试加密序列化）

**editable source 配置**

```toml
# 文件: libs/langgraph/pyproject.toml

[tool.uv.sources]
langgraph-prebuilt = { path = "../prebuilt", editable = true }
langgraph-checkpoint = { path = "../checkpoint", editable = true }
langgraph-checkpoint-sqlite = { path = "../checkpoint-sqlite", editable = true }
langgraph-checkpoint-postgres = { path = "../checkpoint-postgres", editable = true }
langgraph-sdk = { path = "../sdk-py", editable = true }
langgraph-cli = { path = "../cli", editable = true }
```

> **源码位置**：`libs/langgraph/pyproject.toml` 第 83-89 行

这段配置是 Monorepo 开发的关键。`[tool.uv.sources]` 告诉 `uv`（包管理器）在
开发环境中使用本地路径的可编辑安装，而非从 PyPI 下载。这意味着开发者修改
`checkpoint` 包的代码后，`langgraph` 包立即可以看到变更，无需重新安装。


### 2.2.2 `langgraph-checkpoint`：Checkpoint 基础接口

> "Library with base interfaces for LangGraph checkpoint savers."

| 属性 | 值 |
|------|------|
| 包名 | `langgraph-checkpoint` |
| 版本 | 4.0.1 |
| 目录 | `libs/checkpoint/` |
| Python | >=3.10 |

**pyproject.toml 依赖分析**

```toml
# 文件: libs/checkpoint/pyproject.toml

[project]
name = "langgraph-checkpoint"
version = "4.0.1"
description = "Library with base interfaces for LangGraph checkpoint savers."

dependencies = [
    "langchain-core>=0.2.38",
    "ormsgpack>=1.12.0",
]
```

> **源码位置**：`libs/checkpoint/pyproject.toml` 第 5-17 行

仅有两个运行时依赖：

| 依赖 | 版本约束 | 用途 |
|------|---------|------|
| `langchain-core` | >=0.2.38 | `RunnableConfig` 类型、序列化支持 |
| `ormsgpack` | >=1.12.0 | MessagePack 序列化（高性能二进制格式） |

`ormsgpack` 是 `msgpack` 的 Rust 实现绑定，性能比纯 Python 的 `msgpack` 快数倍。
选择 MessagePack 而非 JSON 作为默认序列化格式，是出于性能和紧凑性考虑——Agent
状态可能包含大量数据（如长消息列表），序列化性能直接影响每步执行的开销。

**包的内部结构**

```
libs/checkpoint/langgraph/checkpoint/
├── base/
│   ├── __init__.py     # BaseCheckpointSaver, Checkpoint, CheckpointMetadata
│   ├── id.py           # UUID v6 生成（单调递增 ID）
│   └── py.typed
├── memory/
│   ├── __init__.py     # InMemorySaver（内存 Checkpoint 实现）
│   └── py.typed
└── serde/
    ├── __init__.py     # 序列化注册表
    ├── base.py         # SerializerProtocol
    ├── jsonplus.py     # JsonPlusSerializer（增强 JSON）
    ├── encrypted.py    # EncryptedSerializer（加密序列化）
    ├── _msgpack.py     # MessagePack 内部实现
    ├── event_hooks.py  # 序列化事件钩子
    ├── types.py        # 序列化类型常量（ERROR, INTERRUPT, RESUME, SCHEDULED）
    └── py.typed
```

注意这个包使用了 **namespace package** 模式——包名是 `langgraph-checkpoint`，
但 Python 包路径是 `langgraph.checkpoint`。这通过 `[tool.hatch.build.targets.wheel]`
配置实现：

```toml
# 文件: libs/checkpoint/pyproject.toml

[tool.hatch.build.targets.wheel]
include = ["langgraph"]
```

> **源码位置**：`libs/checkpoint/pyproject.toml` 第 48-49 行

这意味着 `langgraph-checkpoint` 安装后的导入路径是 `from langgraph.checkpoint import ...`，
而不是 `from langgraph_checkpoint import ...`。这个命名空间共享是 LangGraph 生态
的一个重要设计选择——所有 checkpoint 实现（内存、SQLite、PostgreSQL）都共享
`langgraph.checkpoint` 命名空间。

**核心抽象**

checkpoint 包定义了三个核心抽象：

1. **`Checkpoint`**（TypedDict）：状态快照数据结构
2. **`CheckpointMetadata`**（TypedDict）：快照元数据
3. **`BaseCheckpointSaver`**（ABC）：存储接口基类

以及两个关键的序列化组件：

4. **`SerializerProtocol`**：序列化器协议
5. **`JsonPlusSerializer`**：增强 JSON 序列化器（支持 datetime、bytes、Decimal 等）

这个包是整个 checkpoint 体系的**基石**——所有具体存储实现都依赖它。


### 2.2.3 `langgraph-checkpoint-postgres`：PostgreSQL 实现

> "Library with a Postgres implementation of LangGraph checkpoint saver."

| 属性 | 值 |
|------|------|
| 包名 | `langgraph-checkpoint-postgres` |
| 版本 | 3.0.4 |
| 目录 | `libs/checkpoint-postgres/` |
| Python | >=3.10 |

**pyproject.toml 依赖分析**

```toml
# 文件: libs/checkpoint-postgres/pyproject.toml

[project]
name = "langgraph-checkpoint-postgres"
version = "3.0.4"
description = "Library with a Postgres implementation of LangGraph checkpoint saver."

dependencies = [
    "langgraph-checkpoint>=2.1.2,<5.0.0",
    "orjson>=3.11.5",
    "psycopg>=3.2.0",
    "psycopg-pool>=3.2.0",
]
```

> **源码位置**：`libs/checkpoint-postgres/pyproject.toml` 第 5-19 行

| 依赖 | 版本约束 | 用途 |
|------|---------|------|
| `langgraph-checkpoint` | >=2.1.2,<5.0.0 | Checkpoint 基础接口 |
| `orjson` | >=3.11.5 | 高性能 JSON 序列化（Rust 实现） |
| `psycopg` | >=3.2.0 | PostgreSQL 驱动（psycopg v3） |
| `psycopg-pool` | >=3.2.0 | 连接池 |

注意使用了 `psycopg` v3（而非传统的 `psycopg2`）——这是 Python PostgreSQL 驱动
的新一代实现，原生支持异步操作。`orjson` 是 Rust 实现的 JSON 库，用于 JSONB
列的高效序列化。

**开发依赖的 editable source**

```toml
# 文件: libs/checkpoint-postgres/pyproject.toml

[tool.uv.sources]
langgraph-checkpoint = { path = "../checkpoint", editable = true }
```

> **源码位置**：`libs/checkpoint-postgres/pyproject.toml` 第 50-51 行

这保证了在开发环境中，`checkpoint-postgres` 总是使用本地的 `checkpoint` 包。

**包的内部结构**

```
libs/checkpoint-postgres/langgraph/checkpoint/postgres/
├── __init__.py         # PostgresSaver（同步 + 异步入口）
├── base.py             # 基础实现和 SQL 语句
├── _internal.py        # 内部工具函数
├── _ainternal.py       # 异步内部工具
├── aio.py              # AsyncPostgresSaver（异步实现）
├── shallow.py          # ShallowPostgresSaver（浅层存储模式）
└── py.typed
```

`shallow.py` 中的 `ShallowPostgresSaver` 是一个有趣的优化变体——它只保存最新的
checkpoint（而非完整历史），适用于不需要"时间旅行"的场景，可以显著减少存储空间。


### 2.2.4 `langgraph-checkpoint-sqlite`：SQLite 实现

> "Library with a SQLite implementation of LangGraph checkpoint saver."

| 属性 | 值 |
|------|------|
| 包名 | `langgraph-checkpoint-sqlite` |
| 版本 | 3.0.3 |
| 目录 | `libs/checkpoint-sqlite/` |
| Python | >=3.10 |

**pyproject.toml 依赖分析**

```toml
# 文件: libs/checkpoint-sqlite/pyproject.toml

[project]
name = "langgraph-checkpoint-sqlite"
version = "3.0.3"
description = "Library with a SQLite implementation of LangGraph checkpoint saver."

dependencies = [
    "langgraph-checkpoint>=3,<5.0.0",
    "aiosqlite>=0.20",
    "sqlite-vec>=0.1.6",
]
```

> **源码位置**：`libs/checkpoint-sqlite/pyproject.toml` 第 5-18 行

| 依赖 | 版本约束 | 用途 |
|------|---------|------|
| `langgraph-checkpoint` | >=3,<5.0.0 | Checkpoint 基础接口 |
| `aiosqlite` | >=0.20 | SQLite 异步驱动 |
| `sqlite-vec` | >=0.1.6 | SQLite 向量扩展 |

`sqlite-vec` 的引入暗示了 SQLite 实现可能支持某种形式的向量存储或搜索功能，
这在 Agent 记忆场景中是有用的（例如，根据语义相似度检索历史上下文）。

**包的内部结构**

```
libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/
├── __init__.py     # SqliteSaver（同步实现）
├── aio.py          # AsyncSqliteSaver（异步实现）
├── utils.py        # 工具函数
└── py.typed
```

结构简洁，同步和异步各一个实现文件。SQLite 实现适合：
- 本地开发和测试
- 单机部署的轻量场景
- 不需要多进程并发写入的场景


### 2.2.5 `langgraph-checkpoint-conformance`：一致性测试套件

> "Conformance test suite for LangGraph checkpointer implementations."

| 属性 | 值 |
|------|------|
| 包名 | `langgraph-checkpoint-conformance` |
| 版本 | 0.0.1 |
| 目录 | `libs/checkpoint-conformance/` |
| Python | >=3.10 |

```toml
# 文件: libs/checkpoint-conformance/pyproject.toml

[project]
name = "langgraph-checkpoint-conformance"
version = "0.0.1"
description = "Conformance test suite for LangGraph checkpointer implementations."

dependencies = [
    "langgraph-checkpoint>=2.0.0",
]
```

> **源码位置**：`libs/checkpoint-conformance/pyproject.toml` 第 5-15 行

这个包不面向最终用户发布（版本仍为 0.0.1），它的职责是为**第三方 Checkpoint 实现**
提供一致性测试。如果你想实现自己的 Checkpoint 存储后端（比如基于 Redis 或
MongoDB），可以使用这个测试套件验证实现的正确性。

这体现了 LangGraph 的开放设计理念——Checkpoint 接口是可扩展的，而一致性测试
保证了所有实现的行为一致。


### 2.2.6 `langgraph-prebuilt`：预制高层组件

> "Library with high-level APIs for creating and executing LangGraph agents
> and tools."

| 属性 | 值 |
|------|------|
| 包名 | `langgraph-prebuilt` |
| 版本 | 1.0.8 |
| 目录 | `libs/prebuilt/` |
| Python | >=3.10 |

**pyproject.toml 依赖分析**

```toml
# 文件: libs/prebuilt/pyproject.toml

[project]
name = "langgraph-prebuilt"
version = "1.0.8"
description = "Library with high-level APIs for creating and executing LangGraph agents and tools."

dependencies = [
    "langgraph-checkpoint>=2.1.0,<5.0.0",
    "langchain-core>=1.0.0",
]
```

> **源码位置**：`libs/prebuilt/pyproject.toml` 第 5-29 行

| 依赖 | 版本约束 | 用途 |
|------|---------|------|
| `langgraph-checkpoint` | >=2.1.0,<5.0.0 | Checkpoint 接口（状态管理） |
| `langchain-core` | >=1.0.0 | LLM 模型接口、Tool 接口 |

注意 `prebuilt` 包**不直接依赖 `langgraph` 核心包**。它的运行时依赖只有
`langgraph-checkpoint` 和 `langchain-core`。但在测试环境中，它需要 `langgraph`
来实际编译和运行图：

```toml
# 文件: libs/prebuilt/pyproject.toml

[dependency-groups]
test = [
    "pytest",
    "pytest-asyncio",
    "pytest-mock",
    "pytest-watcher",
    "langchain-core",
    "langgraph",                    # 仅测试需要
    "langgraph-checkpoint",
    "langgraph-checkpoint-sqlite",
    "langgraph-checkpoint-postgres",
    "syrupy",
    "psycopg-binary",
]
```

> **源码位置**：`libs/prebuilt/pyproject.toml` 第 37-50 行

**包的内部结构与公共 API**

```
libs/prebuilt/langgraph/prebuilt/
├── __init__.py              # 公共 API 导出
├── chat_agent_executor.py   # create_react_agent
├── tool_node.py             # ToolNode, tools_condition, InjectedState, InjectedStore
├── tool_validator.py        # ValidationNode
├── interrupt.py             # 中断相关工具
└── py.typed
```

公共 API 导出：

```python
# 文件: libs/prebuilt/langgraph/prebuilt/__init__.py

"""langgraph.prebuilt exposes a higher-level API for creating and
executing agents and tools."""

from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.prebuilt.tool_node import (
    InjectedState,
    InjectedStore,
    ToolNode,
    ToolRuntime,
    tools_condition,
)
from langgraph.prebuilt.tool_validator import ValidationNode

__all__ = [
    "create_react_agent",
    "ToolNode",
    "tools_condition",
    "ValidationNode",
    "InjectedState",
    "InjectedStore",
    "ToolRuntime",
]
```

> **源码位置**：`libs/prebuilt/langgraph/prebuilt/__init__.py`

`prebuilt` 包的核心组件：

| 组件 | 文件 | 用途 |
|------|------|------|
| `create_react_agent` | `chat_agent_executor.py` | 创建 ReAct Agent（思考-行动循环） |
| `ToolNode` | `tool_node.py` | 工具调用节点（批量执行 LLM 选择的工具） |
| `tools_condition` | `tool_node.py` | 条件路由（有工具调用 -> ToolNode，否则 -> END） |
| `ValidationNode` | `tool_validator.py` | 工具参数验证节点 |
| `InjectedState` | `tool_node.py` | 工具函数的状态注入标注 |
| `InjectedStore` | `tool_node.py` | 工具函数的 Store 注入标注 |
| `ToolRuntime` | `tool_node.py` | 工具函数的 Runtime 注入标注 |

`create_react_agent` 是最常用的入口——它将 LLM、工具列表和可选配置组合成一个
完整的 ReAct Agent 图。这是 LangGraph 的"高层 API"，底层使用 `StateGraph`
构建。

`InjectedState` 和 `InjectedStore` 是依赖注入标注，允许工具函数声明自己需要
访问图状态或持久化存储。这种设计使得工具函数既可以独立测试，又可以在 Agent
上下文中获得额外能力。


### 2.2.7 `langgraph-cli`：命令行工具

> "CLI for interacting with LangGraph API"

| 属性 | 值 |
|------|------|
| 包名 | `langgraph-cli` |
| 版本 | 0.4.15（动态） |
| 目录 | `libs/cli/` |
| Python | >=3.10 |

**pyproject.toml 依赖分析**

```toml
# 文件: libs/cli/pyproject.toml

[project]
name = "langgraph-cli"
dynamic = ["version"]
description = "CLI for interacting with LangGraph API"

dependencies = [
    "click>=8.1.7",
    "httpx>=0.24.0",
    "langgraph-sdk>=0.1.0 ; python_version >= '3.11'",
    "python-dotenv>=0.8.0",
]

[tool.hatch.version]
path = "langgraph_cli/__init__.py"
```

> **源码位置**：`libs/cli/pyproject.toml` 第 5-21 行

| 依赖 | 版本约束 | 用途 |
|------|---------|------|
| `click` | >=8.1.7 | 命令行框架 |
| `httpx` | >=0.24.0 | HTTP 客户端 |
| `langgraph-sdk` | >=0.1.0 (Python >= 3.11) | SDK 交互 |
| `python-dotenv` | >=0.8.0 | 环境变量管理 |

注意两个设计细节：

1. **动态版本**：`dynamic = ["version"]` 配合 `[tool.hatch.version]` 从源码文件
   读取版本号（`langgraph_cli/__init__.py` 中的 `__version__ = "0.4.15"`）。
   这避免了在发布时需要同步两处版本号。

2. **条件依赖**：`langgraph-sdk` 仅在 Python >= 3.11 时引入。

**可选依赖（inmem）**

```toml
# 文件: libs/cli/pyproject.toml

[project.optional-dependencies]
inmem = [
    "langgraph-api>=0.5.35,<0.8.0 ; python_version >= '3.11'",
    "langgraph-runtime-inmem>=0.7 ; python_version >= '3.11'",
]
```

> **源码位置**：`libs/cli/pyproject.toml` 第 22-26 行

`inmem` 可选依赖引入了 `langgraph-api` 和 `langgraph-runtime-inmem`——这些是
LangGraph Platform 的运行时组件，允许在本地以内存模式运行 LangGraph API 服务器
进行开发和调试。安装方式为 `pip install "langgraph-cli[inmem]"`。

**命令行入口**

```toml
# 文件: libs/cli/pyproject.toml

[project.scripts]
langgraph = "langgraph_cli.cli:cli"
```

> **源码位置**：`libs/cli/pyproject.toml` 第 34-35 行

安装后，用户可以通过 `langgraph` 命令直接使用 CLI。

**包的内部结构**

```
libs/cli/langgraph_cli/
├── __init__.py       # 版本号（__version__ = "0.4.15"）
├── __main__.py       # python -m langgraph_cli 入口
├── cli.py            # click 命令定义（主入口）
├── config.py         # LangGraph 项目配置加载
├── constants.py      # CLI 常量
├── docker.py         # Docker 镜像构建和管理
├── exec.py           # 执行管理
├── host_backend.py   # 主机后端（本地运行时）
├── progress.py       # 进度条和日志
├── schemas.py        # 配置文件 schema 验证
├── templates.py      # 项目模板生成
├── analytics.py      # 使用统计
├── util.py           # 通用工具函数
└── version.py        # 版本管理
```

注意 CLI 包使用的是 `langgraph_cli`（下划线分隔）而非 `langgraph.cli`（点分隔）。
这与 checkpoint 包不同——CLI 包没有使用 namespace package 模式，因为它的功能与
核心 LangGraph 运行时是独立的。


### 2.2.8 `langgraph-sdk`：Python SDK

> "SDK for interacting with LangGraph API"

| 属性 | 值 |
|------|------|
| 包名 | `langgraph-sdk` |
| 版本 | 0.3.10（动态） |
| 目录 | `libs/sdk-py/` |
| Python | >=3.10 |

**pyproject.toml 依赖分析**

```toml
# 文件: libs/sdk-py/pyproject.toml

[project]
name = "langgraph-sdk"
dynamic = ["version"]
description = "SDK for interacting with LangGraph API"

dependencies = ["httpx>=0.25.2", "orjson>=3.11.5"]

[tool.hatch.version]
path = "langgraph_sdk/__init__.py"
```

> **源码位置**：`libs/sdk-py/pyproject.toml` 第 5-18 行

| 依赖 | 版本约束 | 用途 |
|------|---------|------|
| `httpx` | >=0.25.2 | HTTP 客户端（同步 + 异步） |
| `orjson` | >=3.11.5 | 高性能 JSON 序列化（Rust 实现） |

SDK 是整个 LangGraph 生态中**依赖最轻量的包**——仅有两个外部依赖，且都不依赖
LangChain 的任何组件。这是有意为之的设计：SDK 用于客户端场景（如 Web 应用后端、
移动端 API 调用），不需要也不应该引入沉重的 ML 框架依赖。

**公共 API**

```python
# 文件: libs/sdk-py/langgraph_sdk/__init__.py

from langgraph_sdk.auth import Auth
from langgraph_sdk.client import get_client, get_sync_client
from langgraph_sdk.encryption import Encryption
from langgraph_sdk.encryption.types import EncryptionContext

__version__ = "0.3.10"

__all__ = [
    "Auth",
    "Encryption",
    "EncryptionContext",
    "get_client",
    "get_sync_client",
]
```

> **源码位置**：`libs/sdk-py/langgraph_sdk/__init__.py`

最常用的入口是 `get_client()` 和 `get_sync_client()`——分别返回异步和同步的
HTTP 客户端。

**包的内部结构**

```
libs/sdk-py/langgraph_sdk/
├── __init__.py       # 公共 API（get_client, get_sync_client, Auth, Encryption）
├── client.py         # 客户端工厂（get_client, get_sync_client）
├── schema.py         # 数据模型（Thread, Run, Assistant 等）
├── errors.py         # 错误定义
├── sse.py            # Server-Sent Events 解析
├── runtime.py        # 运行时信息
├── cache.py          # 缓存
├── _async/           # 异步客户端实现
├── _sync/            # 同步客户端实现
├── _shared/          # 异步/同步共享的工具代码
├── auth/             # 认证模块
├── encryption/       # 端到端加密
└── py.typed
```

SDK 的代码量（约 12,380 行）在所有包中排名第二，这主要是因为同步和异步客户端
各有一套完整实现，加上 schema 定义、SSE 解析、认证和加密等功能。

SDK 和 CLI 一样使用 `langgraph_sdk`（下划线分隔）的包名，不使用 namespace
package 模式。


### 2.2.9 `sdk-js`：JavaScript SDK（外部仓库）

`libs/sdk-js/` 目录只包含一个 `README.md`，指向独立的
[langgraphjs](https://github.com/langchain-ai/langgraphjs) 仓库。JavaScript SDK
在本书中不做详细分析。


## 2.3 依赖关系图

### 2.3.1 运行时依赖全景图

根据各 `pyproject.toml` 中的 `dependencies` 字段，可以绘制出完整的运行时依赖图：

```
                        langchain-core
                       ┌──────┴──────┐
                       │             │
                       ▼             ▼
                  ormsgpack      langgraph-checkpoint ◄───────────────┐
                       │         │        ▲       ▲                  │
                       │         │        │       │                  │
                       ▼         │   aiosqlite  psycopg             │
               langgraph-checkpoint     │    psycopg-pool           │
                                 │      │       │                   │
                                 │      ▼       ▼                   │
                                 │  checkpoint  checkpoint          │
                                 │  -sqlite     -postgres           │
                                 │                                  │
                                 │                                  │
                                 ▼                                  │
                          langgraph-prebuilt ────────────────────────┘
                                 │
                                 │   langgraph-sdk ◄─── httpx, orjson
                                 │         │
                                 ▼         ▼
                            langgraph ◄────┘
                              │       │
                              ▼       ▼
                          xxhash   pydantic


  langgraph-cli ◄─── click, httpx, python-dotenv
       │
       └──► langgraph-sdk（条件依赖，Python >= 3.11）
```

### 2.3.2 简化依赖链

如果只关注 LangGraph 自身的包间依赖，关系可以大幅简化：

```
langgraph-checkpoint          langgraph-sdk
    ▲    ▲    ▲                    ▲    ▲
    │    │    │                    │    │
    │    │    └── checkpoint-postgres  │
    │    │                             │
    │    └── checkpoint-sqlite         │
    │                                  │
    └── langgraph-prebuilt             │
              │                        │
              ▼                        │
         langgraph ◄───────────────────┘

  langgraph-cli ──────► langgraph-sdk (条件)
```

核心的依赖链路是：

```
checkpoint ──► langgraph ◄── sdk
                  ▲
                  │
               prebuilt
```

### 2.3.3 依赖层次分析

从依赖图可以清楚地看到**分层结构**：

**第 0 层：外部基础依赖**

```
langchain-core, ormsgpack, httpx, orjson, click, pydantic, xxhash,
psycopg, psycopg-pool, aiosqlite, sqlite-vec, python-dotenv
```

这些都是 LangGraph 不控制的外部包。

**第 1 层：基础接口**

```
langgraph-checkpoint  [依赖: langchain-core, ormsgpack]
langgraph-sdk         [依赖: httpx, orjson]
```

这两个包是独立的基础组件，互不依赖。

**第 2 层：具体实现**

```
langgraph-checkpoint-sqlite    [依赖: 第1层 checkpoint]
langgraph-checkpoint-postgres  [依赖: 第1层 checkpoint]
langgraph-prebuilt             [依赖: 第1层 checkpoint + langchain-core]
```

**第 3 层：核心框架**

```
langgraph  [依赖: 第1层 checkpoint + sdk, 第2层 prebuilt]
```

**独立层：工具链**

```
langgraph-cli  [依赖: 第1层 sdk + 外部工具]
langgraph-checkpoint-conformance  [依赖: 第1层 checkpoint]
```

### 2.3.4 反向依赖（谁依赖了我）

从"被依赖"的角度来看：

| 包 | 运行时被谁依赖 |
|------|---------|
| `langgraph-checkpoint` | langgraph, prebuilt, checkpoint-postgres, checkpoint-sqlite, conformance（5 个包） |
| `langchain-core` | langgraph, checkpoint, prebuilt（3 个包） |
| `langgraph-sdk` | langgraph, cli（2 个包） |
| `langgraph-prebuilt` | langgraph（1 个包） |
| `langgraph` | (顶层包，不被其他 LangGraph 包在运行时依赖) |

`langgraph-checkpoint` 是被依赖最多的包（5 个包依赖它），这证实了它作为
"基础接口层"的核心定位。

### 2.3.5 一个值得注意的"循环"

细心的读者可能注意到一个有趣的依赖关系：

- `langgraph` 依赖 `langgraph-prebuilt`
- `langgraph-prebuilt` 在**测试**中依赖 `langgraph`

但这**不是**运行时循环依赖——`prebuilt` 的 `pyproject.toml` 中，`langgraph` 只
出现在 `[dependency-groups] test` 中，不在 `dependencies` 中。这意味着：

- **发布时**：`prebuilt` 不依赖 `langgraph`，可以独立安装
- **开发/测试时**：`prebuilt` 需要 `langgraph` 来运行集成测试

同样，`langgraph` 的核心代码中实际上并不 `import langgraph.prebuilt`——prebuilt
包被 `langgraph` 的 `pyproject.toml` 列为依赖，主要是为了方便用户安装
（`pip install langgraph` 自动获得 prebuilt 组件）。


## 2.4 构建系统统一配置

### 2.4.1 Hatchling 构建后端

所有包统一使用 `hatchling` 作为构建后端：

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

`hatchling` 是 `hatch` 项目管理工具的构建后端，相比 `setuptools` 更现代化，
支持动态版本、灵活的文件包含/排除规则，配置更简洁。

### 2.4.2 Wheel 包含规则

各包通过 `[tool.hatch.build.targets.wheel]` 精确控制发布内容：

```toml
# langgraph 核心包
[tool.hatch.build.targets.wheel]
packages = ["langgraph"]

# checkpoint 及其实现（namespace package）
[tool.hatch.build.targets.wheel]
include = ["langgraph"]

# CLI（独立命名空间）
[tool.hatch.build.targets.wheel]
include = ["langgraph_cli"]

# SDK（独立命名空间）
[tool.hatch.build.targets.wheel]
include = ["langgraph_sdk"]
```

注意 `packages` 和 `include` 的区别：
- `packages = ["langgraph"]`：将 `langgraph` 目录作为顶层包
- `include = ["langgraph"]`：包含 `langgraph` 目录下的所有内容（用于 namespace package）

对于使用 namespace package 模式的包（checkpoint、checkpoint-postgres、
checkpoint-sqlite、prebuilt），安装后它们的代码会合并到同一个 `langgraph/`
命名空间下：

```
site-packages/langgraph/
├── __init__.py          # 来自 langgraph 核心包
├── graph/               # 来自 langgraph 核心包
├── pregel/              # 来自 langgraph 核心包
├── checkpoint/          # 来自 langgraph-checkpoint
│   ├── base/            #   来自 langgraph-checkpoint
│   ├── memory/          #   来自 langgraph-checkpoint
│   ├── serde/           #   来自 langgraph-checkpoint
│   ├── postgres/        #   来自 langgraph-checkpoint-postgres
│   └── sqlite/          #   来自 langgraph-checkpoint-sqlite
└── prebuilt/            # 来自 langgraph-prebuilt
```

这种设计使得用户可以统一使用 `from langgraph.checkpoint.postgres import ...` 的
导入路径，而不需要关心底层是哪个 PyPI 包提供的。


### 2.4.3 uv 包管理器

LangGraph 项目在开发中使用 `uv` 作为包管理器。`uv` 是 Rust 编写的超快 Python
包管理器（由 Astral 开发，也是 Ruff 的开发团队），从各 `pyproject.toml` 中的
`[tool.uv]` 和 `[tool.uv.sources]` 配置可以看到其使用方式。

每个子包都通过 `[tool.uv.sources]` 配置了对其他包的本地可编辑引用。以核心包
`langgraph` 为例：

```toml
# 文件: libs/langgraph/pyproject.toml

[tool.uv.sources]
langgraph-prebuilt = { path = "../prebuilt", editable = true }
langgraph-checkpoint = { path = "../checkpoint", editable = true }
langgraph-checkpoint-sqlite = { path = "../checkpoint-sqlite", editable = true }
langgraph-checkpoint-postgres = { path = "../checkpoint-postgres", editable = true }
langgraph-sdk = { path = "../sdk-py", editable = true }
langgraph-cli = { path = "../cli", editable = true }
```

> **源码位置**：`libs/langgraph/pyproject.toml` 第 83-89 行

这意味着开发者只需在某个包的目录下运行 `uv sync`，就能自动解析并安装所有
本地依赖为可编辑模式。


### 2.4.4 代码质量工具

所有包统一使用以下代码质量工具。

**1. Ruff（Linter + Formatter）**

```toml
# 典型配置（来自 libs/langgraph/pyproject.toml）

[tool.ruff]
lint.select = [ "E", "F", "I", "TID251", "UP" ]
lint.ignore = [ "E501" ]
line-length = 88
indent-width = 4
target-version = "py310"
```

> **源码位置**：`libs/langgraph/pyproject.toml` 第 91-97 行

启用的规则集：
- `E`：pycodestyle 错误
- `F`：Pyflakes（未使用导入等）
- `I`：isort（导入排序）
- `TID251`：禁止特定导入
- `UP`：pyupgrade（Python 版本兼容性）

所有包忽略 `E501`（行长度限制），因为使用 formatter 自动处理。

特别值得注意的是 `TID251` 规则的自定义配置：

```toml
# 文件: libs/langgraph/pyproject.toml

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.TypedDict".msg = "Use typing_extensions.TypedDict instead."
```

> **源码位置**：`libs/langgraph/pyproject.toml` 第 99-100 行

这强制项目中使用 `typing_extensions.TypedDict` 而非标准库的 `typing.TypedDict`。
原因是 `typing_extensions` 版本提供了更多特性（如 `Required`、`NotRequired`、
`Unpack` 等），且跨 Python 版本行为一致。

SDK 包的 Ruff 配置更严格，启用了更多规则：

```toml
# 文件: libs/sdk-py/pyproject.toml

[tool.ruff.lint]
select = [
  "E",      # pycodestyle errors
  "F",      # pyflakes
  "I",      # isort (import sorting)
  "ARG",    # unused arguments
  "TID251", # banned imports
  "TID252", # banned relative imports
  "T20",    # print statements
  "UP",     # pyupgrade
  "B",      # flake8-bugbear (common bugs)
  "SIM",    # flake8-simplify (code simplification)
  "RUF",    # ruff-specific rules
  "S101",   # flake8-bandit: use of assert
]
```

> **源码位置**：`libs/sdk-py/pyproject.toml` 第 62-75 行

SDK 额外禁止了 `print` 语句（`T20`）、检查未使用参数（`ARG`）、禁止 `assert`
（`S101`，安全考虑）等——这反映了 SDK 作为面向客户端的库需要更高的代码质量标准。

**2. Mypy（类型检查）**

```toml
# 典型配置

[tool.mypy]
disallow_untyped_defs = "True"
explicit_package_bases = "True"
warn_no_return = "False"
warn_unused_ignores = "True"
warn_redundant_casts = "True"
allow_redefinition = "True"
disable_error_code = "typeddict-item, return-value, override, has-type"
```

> **源码位置**：`libs/langgraph/pyproject.toml` 第 102-110 行

`disallow_untyped_defs = True` 表示所有函数必须有类型注解——这是一个严格的要求，
保证了代码库的类型安全性。被禁用的错误码（`typeddict-item`、`return-value` 等）
主要是因为 LangGraph 的某些动态模式难以用 mypy 静态分析。

**3. Pytest 配置**

```toml
# 核心包的 pytest 配置

[tool.pytest.ini_options]
addopts = "--full-trace --strict-markers --strict-config --durations=5
           --snapshot-warn-unused"
```

> **源码位置**：`libs/langgraph/pyproject.toml` 第 123-124 行

- `--strict-markers`：未注册的 marker 视为错误
- `--strict-config`：配置错误视为错误
- `--durations=5`：显示最慢的 5 个测试
- `--snapshot-warn-unused`：警告未使用的快照（配合 `syrupy`）

大多数子包还配置了 `asyncio_mode = "auto"`，使得所有 `async def test_*` 函数
自动使用 `pytest-asyncio` 运行。


### 2.4.5 依赖组（Dependency Groups）

LangGraph 使用 PEP 735 提案的 dependency groups 特性组织开发依赖：

```toml
[dependency-groups]
test = [
    "pytest",
    "pytest-cov",
    ...
]
lint = [
    "mypy",
    "ruff",
    ...
]
dev = [
    {include-group = "test"},
    {include-group = "lint"},
    "jupyter",
]
```

这种结构允许开发者根据需要选择性安装：
- `uv sync --group test`：只安装测试依赖
- `uv sync --group lint`：只安装 lint 依赖
- `uv sync --group dev`：安装所有开发依赖（测试 + lint + 其他）

`{include-group = "test"}` 语法表示包含另一个组的所有依赖，类似于继承。


## 2.5 Namespace Package 机制详解

### 2.5.1 什么是 Namespace Package

LangGraph 的 checkpoint 和 prebuilt 包使用了 Python 的 **implicit namespace
package**（PEP 420）机制。这允许多个独立的 PyPI 包向同一个 Python 命名空间贡献
代码。

具体来说：

- `langgraph-checkpoint` 提供 `langgraph/checkpoint/base/` 和 `langgraph/checkpoint/serde/`
- `langgraph-checkpoint-postgres` 提供 `langgraph/checkpoint/postgres/`
- `langgraph-checkpoint-sqlite` 提供 `langgraph/checkpoint/sqlite/`
- `langgraph-prebuilt` 提供 `langgraph/prebuilt/`

这些包安装后，Python 的 import 系统会自动合并它们到同一个 `langgraph` 命名空间。

### 2.5.2 py.typed 标记

每个子包目录中都包含 `py.typed` 文件，这是 PEP 561 定义的标记文件，表示该包
支持类型检查。这保证了 mypy 等工具可以正确解析跨包的类型信息。

### 2.5.3 命名空间选择策略

LangGraph 中的包使用了两种不同的命名空间策略：

| 策略 | 包 | Python 包名 | 导入方式 |
|------|------|-----------|---------|
| Namespace package | checkpoint, postgres, sqlite, prebuilt | `langgraph.*` | `from langgraph.checkpoint.postgres import ...` |
| 独立包名 | cli, sdk-py | `langgraph_cli`, `langgraph_sdk` | `from langgraph_sdk import ...` |

选择的依据是：
- **运行时概念相关**的包（checkpoint、prebuilt）使用 namespace package，提供统一的导入体验
- **工具/客户端**类型的包（cli、sdk）使用独立包名，避免与核心运行时混淆

### 2.5.4 潜在风险与规避

Namespace package 模式有一个潜在风险：如果两个包向同一个目录安装了不同版本的
`__init__.py`，可能导致冲突。LangGraph 通过以下方式规避这个问题：

1. `langgraph/checkpoint/` 目录下没有 `__init__.py`（隐式命名空间）
2. 每个子模块（`base/`、`postgres/`、`sqlite/`）有独立的 `__init__.py`
3. 各包的 wheel 包含规则精确控制了文件范围


## 2.6 版本兼容性矩阵

由于各包独立发布，版本兼容性是一个重要的工程关注点。以下是当前版本的兼容性约束
总结：

| 包 | 对 checkpoint 的版本约束 | 对 langchain-core 的约束 |
|------|------------------------|------------------------|
| `langgraph` 1.1.0 | >=2.1.0,<5.0.0 | >=0.1 |
| `checkpoint-postgres` 3.0.4 | >=2.1.2,<5.0.0 | (间接) |
| `checkpoint-sqlite` 3.0.3 | >=3,<5.0.0 | (间接) |
| `prebuilt` 1.0.8 | >=2.1.0,<5.0.0 | >=1.0.0 |
| `conformance` 0.0.1 | >=2.0.0 | (间接) |

`langgraph-checkpoint` 4.0.1 落在所有约束的交集内（>=3,<5.0.0），保证了当前版本
的完全兼容。

值得注意的是 `checkpoint` 包使用了较高的大版本号（4.x），说明其接口经历了多次
不兼容变更。但消费方的版本约束（如 `>=2.1.0,<5.0.0`）表明，这些变更在 2.x 到
4.x 之间保持了某种程度的向后兼容。`checkpoint-sqlite` 的约束是最严格的
（`>=3,<5.0.0`），说明 3.x 版本引入了 SQLite 实现所需的 API 变更。


## 2.7 如何在本地搭建开发环境

基于对仓库结构的理解，搭建 LangGraph 本地开发环境的步骤如下。

### 2.7.1 前置条件

- Python >= 3.10
- uv 包管理器
- PostgreSQL（如需运行 checkpoint-postgres 测试）

### 2.7.2 安装核心包

```bash
# 克隆仓库
git clone https://github.com/langchain-ai/langgraph.git
cd langgraph

# 安装核心包及其所有本地依赖
cd libs/langgraph
uv sync --group dev
```

`uv sync` 会自动解析 `[tool.uv.sources]` 中的本地路径，以可编辑模式安装所有
依赖包。修改任何一个包的代码后，无需重新安装。

### 2.7.3 安装特定子包

```bash
# 只安装 checkpoint 相关
cd libs/checkpoint
uv sync --group dev

# 安装 prebuilt
cd libs/prebuilt
uv sync --group dev
```

### 2.7.4 运行测试

```bash
# 在核心包目录下
cd libs/langgraph
uv run pytest tests/unit_tests/ -x

# 运行特定子包的测试
cd libs/checkpoint
uv run pytest

cd libs/prebuilt
uv run pytest
```

### 2.7.5 代码质量检查

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format --check .

# 类型检查
uv run mypy langgraph
```

---

## 2.8 本章要点

1. **Monorepo 架构**：LangGraph 将 8 个独立发布的 Python 包 + 1 个内部测试套件
   组织在同一个 Git 仓库的 `libs/` 目录下，通过 `uv` 的 editable source 配置
   实现高效的跨包开发。

2. **核心包 `langgraph`**（1.1.0，约 19,660 行）：运行时核心，包含 StateGraph
   构建器、Pregel 执行引擎、通道系统、Functional API 等。依赖 `langchain-core`
   （Runnable 接口）和 `langgraph-checkpoint`（持久化接口）。

3. **Checkpoint 体系**：
   - `langgraph-checkpoint`（4.0.1）：基础接口和序列化（`BaseCheckpointSaver`、
     `Checkpoint`、`JsonPlusSerializer`），使用 `ormsgpack` 高性能序列化
   - `langgraph-checkpoint-postgres`（3.0.4）：生产级 PostgreSQL 实现（`psycopg` v3）
   - `langgraph-checkpoint-sqlite`（3.0.3）：轻量级 SQLite 实现
   - `langgraph-checkpoint-conformance`（0.0.1）：第三方实现的一致性测试套件

4. **高层组件 `langgraph-prebuilt`**（1.0.8）：`create_react_agent`、`ToolNode`
   等开箱即用的 Agent 构建工具，依赖 `langchain-core` 的模型接口。运行时不依赖
   `langgraph` 核心包。

5. **工具链**：
   - `langgraph-cli`（0.4.15）：命令行工具，基于 `click`，支持 Docker 部署和
     本地内存模式运行
   - `langgraph-sdk`（0.3.10）：Python HTTP 客户端，最轻量（仅依赖 `httpx`、
     `orjson`），无 LangChain 依赖

6. **依赖层次**：
   ```
   checkpoint（基础接口）──► langgraph（核心框架）◄── sdk（独立客户端）
        ▲                        ▲
        │                        │
   postgres / sqlite         prebuilt（高层组件）
   ```

7. **Namespace Package**：checkpoint 系列和 prebuilt 使用 Python namespace package
   机制，共享 `langgraph.*` 命名空间。CLI 和 SDK 使用独立的 `langgraph_cli` /
   `langgraph_sdk` 命名空间。

8. **统一工具链**：所有包使用 `hatchling` 构建、`uv` 管理依赖、`ruff` lint/format、
   `mypy` 类型检查、`pytest` 测试。强制使用 `typing_extensions.TypedDict`，
   目标 Python 3.10+。

9. **代码规模**：总计约 56,370 行 Python 源码（不含测试），其中核心包约 19,660 行
   占 35%，SDK 约 12,380 行占 22%，checkpoint 体系约 16,320 行占 29%。
