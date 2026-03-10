# 第 2 章 Monorepo 结构与模块依赖

LangGraph 采用 monorepo（单一仓库）架构，将多个相互关联的包集中管理。理解仓库结构和模块依赖关系，是深入源码阅读的前提。本章将逐一剖析每个子包的职责、依赖，并绘制出完整的依赖关系图。

---

## 2.1 顶层目录结构

LangGraph 仓库的核心代码集中在 `libs/` 目录下。该目录包含 9 个子目录：

```
libs/
├── langgraph/                  # 核心框架
├── checkpoint/                 # Checkpoint 基础接口
├── checkpoint-postgres/        # Checkpoint PostgreSQL 实现
├── checkpoint-sqlite/          # Checkpoint SQLite 实现
├── checkpoint-conformance/     # Checkpoint 一致性测试套件
├── prebuilt/                   # 预制高层组件（Agent、Tool 等）
├── cli/                        # 命令行工具
├── sdk-py/                     # Python SDK
└── sdk-js/                     # JavaScript SDK（已迁移至独立仓库）
```

其中 `sdk-js` 目录仅保留了一个 `README.md`，指向独立仓库：

> This repository has been moved to [langchain-ai/langgraphjs](https://github.com/langchain-ai/langgraphjs/tree/main/libs/sdk).

因此，实际活跃维护的 Python 包有 **8 个**。下面逐一分析。

## 2.2 各模块职责与依赖详解

### 2.2.1 `langgraph-checkpoint`：Checkpoint 基础接口

**路径**：`libs/checkpoint/`

**定位**：定义 Checkpoint 的核心抽象，是整个持久化体系的基石。

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

**关键观察**：

- **无 LangGraph 依赖**：这是整个依赖树的叶子节点，不依赖任何其他 LangGraph 包。
- **依赖 `langchain-core`**：使用其基础类型系统（`Serializable` 等）。
- **依赖 `ormsgpack`**：用于 Checkpoint 数据的高性能 MessagePack 序列化。这意味着 LangGraph 选择了二进制序列化而非 JSON，优先考虑性能。

此包提供的核心类是 `BaseCheckpointSaver`：

```python
# 文件: libs/checkpoint/langgraph/checkpoint/base/__init__.py

class BaseCheckpointSaver(Generic[V]):
    """Base class for creating a graph checkpointer.

    Checkpointers allow LangGraph agents to persist their state
    within and across multiple interactions.
    """
```

所有具体的 Checkpoint 实现（SQLite、PostgreSQL 等）都继承自这个基类。

### 2.2.2 `langgraph-checkpoint-sqlite`：SQLite 实现

**路径**：`libs/checkpoint-sqlite/`

**定位**：基于 SQLite 的 Checkpoint 存储，适合开发环境和轻量级场景。

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

**关键观察**：

- **依赖 `langgraph-checkpoint`**：实现其定义的 `BaseCheckpointSaver` 接口。
- **依赖 `aiosqlite`**：支持异步 SQLite 操作，这意味着 Checkpoint 的读写不会阻塞事件循环。
- **依赖 `sqlite-vec`**：这是一个 SQLite 的向量搜索扩展，暗示 Checkpoint 存储可能涉及向量化的 memory 检索功能。

### 2.2.3 `langgraph-checkpoint-postgres`：PostgreSQL 实现

**路径**：`libs/checkpoint-postgres/`

**定位**：基于 PostgreSQL 的 Checkpoint 存储，面向生产环境。

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

**关键观察**：

- **依赖 `langgraph-checkpoint`**：与 SQLite 实现一样，实现基类接口。
- **使用 `psycopg` v3**：而非 v2（`psycopg2`），拥抱了 Python PostgreSQL 驱动的最新版本，原生支持异步。
- **`psycopg-pool`**：使用连接池管理数据库连接，这是生产环境必备的性能优化。
- **`orjson`**：高性能 JSON 序列化库，用于部分数据的 JSON 存储。

### 2.2.4 `langgraph-prebuilt`：预制高层组件

**路径**：`libs/prebuilt/`

**定位**：在核心框架之上提供开箱即用的 Agent 和 Tool 调用模式。

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

**关键观察**：

- **依赖 `langgraph-checkpoint`**：需要 Checkpoint 接口来支持有状态的预制 Agent。
- **依赖 `langchain-core`**：使用 LangChain 的工具（Tool）和模型（ChatModel）抽象。
- **不直接依赖 `langgraph` 核心包**：在运行时（runtime）依赖中，`prebuilt` 的 `pyproject.toml` 并没有将 `langgraph` 列为发布依赖。但在开发测试时（`[dependency-groups] test`），它确实使用了 `langgraph`。这是因为 `prebuilt` 提供的是**可以被 `langgraph` 导入使用的组件**，而非直接依赖 `langgraph` 的上层应用。

值得注意的是，在 `langgraph` 核心包的 `pyproject.toml` 中：

```toml
# 文件: libs/langgraph/pyproject.toml

dependencies = [
    ...
    "langgraph-prebuilt>=1.0.8,<1.1.0",
    ...
]
```

`langgraph` 将 `langgraph-prebuilt` 作为自己的依赖——也就是说，安装 `langgraph` 时会自动安装 `prebuilt`，用户可以直接使用预制组件。

### 2.2.5 `langgraph`：核心框架

**路径**：`libs/langgraph/`

**定位**：整个项目的核心，包含图定义（`StateGraph`）、运行时引擎（`Pregel`）、Channel 系统等。

```toml
# 文件: libs/langgraph/pyproject.toml

[project]
name = "langgraph"
version = "1.1.0"
description = "Building stateful, multi-actor applications with LLMs"
dependencies = [
    "langchain-core>=0.1",
    "langgraph-checkpoint>=2.1.0,<5.0.0",
    "langgraph-sdk>=0.3.0,<0.4.0",
    "langgraph-prebuilt>=1.0.8,<1.1.0",
    "xxhash>=3.5.0",
    "pydantic>=2.7.4",
]
```

**关键观察**：

- **依赖 `langgraph-checkpoint`**：核心框架需要 Checkpoint 接口来实现持久执行。
- **依赖 `langgraph-sdk`**：集成 SDK 的类型定义，支持远程调用场景。
- **依赖 `langgraph-prebuilt`**：将预制组件打包为默认安装项。
- **`xxhash`**：一种极快的非加密哈希算法，用于 Channel 状态的快速变更检测。
- **`pydantic>=2.7.4`**：使用 Pydantic v2 进行数据验证和序列化。

核心包的内部模块结构：

```
libs/langgraph/langgraph/
├── _internal/        # 内部实现细节（不属于公开 API）
├── channels/         # Channel 实现（LastValue, Topic 等）
├── func/             # Functional API（entrypoint 装饰器等）
├── graph/            # Graph API（StateGraph, CompiledStateGraph）
├── managed/          # Managed Value（生命周期管理的值）
├── pregel/           # Pregel 运行时引擎
├── utils/            # 工具函数
├── config.py         # 配置相关
├── constants.py      # 公开常量（START, END 等）
├── errors.py         # 异常类型
├── types.py          # 公开类型定义
├── typing.py         # 类型变量
├── runtime.py        # 运行时入口
├── version.py        # 版本信息
└── warnings.py       # 废弃警告
```

这个结构清晰地反映了 LangGraph 的分层设计：`graph/` 负责定义，`pregel/` 负责执行，`channels/` 负责通信。

### 2.2.6 `langgraph-sdk`：Python SDK

**路径**：`libs/sdk-py/`

**定位**：用于与部署后的 LangGraph API 服务端通信的客户端 SDK。

```toml
# 文件: libs/sdk-py/pyproject.toml

[project]
name = "langgraph-sdk"
description = "SDK for interacting with LangGraph API"
dependencies = ["httpx>=0.25.2", "orjson>=3.11.5"]
```

**关键观察**：

- **极少依赖**：仅依赖 `httpx`（HTTP 客户端）和 `orjson`（JSON 序列化）。
- **不依赖任何 LangGraph 包**：SDK 是完全独立的，它通过 HTTP 与 LangGraph 服务端通信，不需要引入核心框架。这是一个重要的设计决策——客户端环境可以非常轻量。
- **版本号动态获取**：通过 `[tool.hatch.version] path = "langgraph_sdk/__init__.py"` 从源码获取版本号。

### 2.2.7 `langgraph-cli`：命令行工具

**路径**：`libs/cli/`

**定位**：提供 `langgraph` 命令行工具，用于本地开发和部署。

```toml
# 文件: libs/cli/pyproject.toml

[project]
name = "langgraph-cli"
description = "CLI for interacting with LangGraph API"
dependencies = [
    "click>=8.1.7",
    "httpx>=0.24.0",
    "langgraph-sdk>=0.1.0 ; python_version >= '3.11'",
    "python-dotenv>=0.8.0",
]

[project.optional-dependencies]
inmem = [
    "langgraph-api>=0.5.35,<0.8.0 ; python_version >= '3.11'",
    "langgraph-runtime-inmem>=0.7 ; python_version >= '3.11'",
]

[project.scripts]
langgraph = "langgraph_cli.cli:cli"
```

**关键观察**：

- **基于 Click 框架**：使用 `click` 构建命令行界面。
- **条件依赖 `langgraph-sdk`**：仅在 Python >= 3.11 时启用 SDK 集成。
- **可选 `inmem` 模式**：安装 `pip install langgraph-cli[inmem]` 可启用内存运行时（`langgraph-runtime-inmem`），无需 Docker 即可本地运行 LangGraph API 服务。
- **注册 CLI 入口**：通过 `[project.scripts]` 注册 `langgraph` 命令。

### 2.2.8 `langgraph-checkpoint-conformance`：一致性测试

**路径**：`libs/checkpoint-conformance/`

**定位**：提供 Checkpoint 实现的一致性测试套件，确保不同后端（SQLite、PostgreSQL 等）行为一致。

```toml
# 文件: libs/checkpoint-conformance/pyproject.toml

[project]
name = "langgraph-checkpoint-conformance"
```

这个包不面向终端用户，主要供 Checkpoint 实现的开发者使用。

## 2.3 依赖关系图

了解了各模块的依赖声明后，我们可以绘制出完整的依赖关系图。以下 ASCII 图展示了运行时（runtime）依赖关系，不包括开发/测试依赖：

```
                        ┌─────────────────┐
                        │  langchain-core  │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
┌──────────────────────┐ ┌──────────────┐ ┌─────────────────┐
│ langgraph-checkpoint │ │   langgraph  │ │langgraph-prebuilt│
│       v4.0.1         │ │    v1.1.0    │ │     v1.0.8      │
│                      │ │              │ │                  │
│  deps: ormsgpack     │ │  deps:       │ │  deps:           │
└──────────┬───────────┘ │  xxhash      │ └────────┬────────┘
           │             │  pydantic    │          │
           │             └──────┬───────┘          │
           │                    │                  │
           │      ┌─────────────┼──────────────────┘
           │      │             │
           │      ▼             │
           │  langgraph 依赖:   │
           │  ┌─ checkpoint ◄───┘
           │  ├─ prebuilt
           │  └─ sdk
           │
     ┌─────┴──────┐          ┌──────────────┐
     │             │          │              │
     ▼             ▼          ▼              │
┌─────────┐ ┌──────────┐ ┌──────────┐      │
│ SQLite  │ │ Postgres │ │   SDK    │      │
│ v3.0.3  │ │  v3.0.4  │ │ (sdk-py) │      │
│         │ │          │ │          │      │
│aiosqlite│ │ psycopg  │ │  httpx   │      │
│sqlite-  │ │ psycopg- │ │  orjson  │      │
│  vec    │ │  pool    │ │          │      │
└─────────┘ │  orjson  │ └──────────┘      │
            └──────────┘                   │
                                           ▼
                                    ┌────────────┐
                                    │    CLI     │
                                    │            │
                                    │  click     │
                                    │  httpx     │
                                    │  dotenv    │
                                    │  sdk (opt) │
                                    └────────────┘
```

用更简洁的方式表达核心依赖链：

```
langchain-core
    │
    ├──► langgraph-checkpoint          （基础接口层）
    │        │
    │        ├──► checkpoint-sqlite    （存储实现）
    │        ├──► checkpoint-postgres   （存储实现）
    │        │
    │        ├──► langgraph-prebuilt   （高层组件）
    │        │
    │        └──► langgraph            （核心框架，同时依赖 prebuilt + sdk）
    │
    └──► langgraph-prebuilt

langgraph-sdk                          （独立，仅 httpx + orjson）

langgraph-cli                          （依赖 click + sdk）
```

## 2.4 依赖关系的设计要点

### 2.4.1 Checkpoint 接口下沉

`langgraph-checkpoint` 被设计为依赖树中最底层的 LangGraph 包。它不依赖 `langgraph` 核心框架，这意味着：

- Checkpoint 实现可以**独立开发和发布**，不受核心框架版本约束。
- 第三方可以轻松编写自定义 Checkpoint 后端（如 Redis、DynamoDB），只需依赖 `langgraph-checkpoint` 即可。
- 版本约束使用宽松范围（`>=2.1.0,<5.0.0`），允许接口的演进而不频繁打破兼容性。

### 2.4.2 SDK 的完全独立

`langgraph-sdk` 仅依赖 `httpx` 和 `orjson`，不引入任何 LangGraph 或 LangChain 包。这个设计使得：

- 客户端环境可以非常轻量（无需安装 PyTorch、transformers 等重依赖）。
- SDK 可以在任何 Python 环境中使用，甚至可以在不安装 LangGraph 的服务器上调用远程 LangGraph API。
- JavaScript SDK 已迁移到独立仓库（`langgraph-ai/langgraphjs`），进一步解耦。

### 2.4.3 Prebuilt 的双向关系

`langgraph-prebuilt` 和 `langgraph` 之间存在一个有趣的关系：

- `langgraph` 的 `pyproject.toml` 将 `langgraph-prebuilt` 列为依赖
- `langgraph-prebuilt` 的 `pyproject.toml` **不**将 `langgraph` 列为运行时依赖

这意味着：安装 `langgraph` 会自动安装 `prebuilt`，但 `prebuilt` 的发布依赖中不包含 `langgraph`。在开发环境中，通过 `[tool.uv.sources]` 配置了本地路径引用：

```toml
# 文件: libs/prebuilt/pyproject.toml

[tool.uv.sources]
langgraph = { path = "../langgraph", editable = true }
langgraph-checkpoint = { path = "../checkpoint", editable = true }
```

这种设计避免了循环依赖：`langgraph` → `prebuilt` → `langgraph` 的循环在运行时不会发生，因为 `prebuilt` 在运行时不强制要求 `langgraph`。

### 2.4.4 CLI 的可选重量级依赖

`langgraph-cli` 提供了 `inmem` 可选依赖组：

```toml
# 文件: libs/cli/pyproject.toml

[project.optional-dependencies]
inmem = [
    "langgraph-api>=0.5.35,<0.8.0 ; python_version >= '3.11'",
    "langgraph-runtime-inmem>=0.7 ; python_version >= '3.11'",
]
```

基础安装（`pip install langgraph-cli`）非常轻量；只有需要本地内存运行时的开发者才需要安装 `inmem` 附加依赖。注意这些依赖（`langgraph-api`、`langgraph-runtime-inmem`）并不在本 monorepo 中，而是在其他仓库或闭源发布的。

## 2.5 构建系统统一

所有 Python 包统一使用以下工具链：

| 工具 | 用途 | 配置位置 |
|---|---|---|
| `hatchling` | 构建后端 | `[build-system]` |
| `uv` | 包管理与虚拟环境 | `[tool.uv]` |
| `ruff` | Linting 与格式化 | `[tool.ruff]` |
| `mypy` | 静态类型检查 | `[tool.mypy]` |
| `pytest` | 测试框架 | `[tool.pytest.ini_options]` |

在 monorepo 开发中，`[tool.uv.sources]` 配置了本地路径引用，使得各包之间可以使用 editable 模式相互引用。以核心包为例：

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

这意味着开发者修改任何一个子包的代码，其他包立即可见，无需重新安装。

## 2.6 命名空间的统一

一个有趣的细节：虽然有多个包，但它们共享 `langgraph` Python 命名空间。观察各包的 `[tool.hatch.build.targets.wheel]` 配置：

| 包 | wheel include |
|---|---|
| `langgraph-checkpoint` | `langgraph` |
| `langgraph-checkpoint-sqlite` | `langgraph` |
| `langgraph-checkpoint-postgres` | `langgraph` |
| `langgraph-prebuilt` | `langgraph` |
| `langgraph` | `langgraph` |

它们都将代码打包在 `langgraph/` 命名空间下。例如：

- `langgraph-checkpoint` 提供 `langgraph.checkpoint.*`
- `langgraph-prebuilt` 提供 `langgraph.prebuilt.*`
- `langgraph` 核心包提供 `langgraph.graph.*`、`langgraph.pregel.*` 等

这种 **namespace package** 的设计让用户感觉所有功能都来自同一个 `langgraph` 包，即使底层是多个独立发布的 PyPI 包。

而 CLI 和 SDK 使用了不同的命名空间：

| 包 | wheel include | 命名空间 |
|---|---|---|
| `langgraph-cli` | `langgraph_cli` | `langgraph_cli.*` |
| `langgraph-sdk` | `langgraph_sdk` | `langgraph_sdk.*` |

这是因为它们属于独立的工具，不需要与核心框架共享命名空间。

---

## 本章要点

1. **Monorepo 架构**：LangGraph 在 `libs/` 下管理 8 个活跃的 Python 包（`sdk-js` 已迁移至独立仓库），统一使用 `hatchling` + `uv` + `ruff` 工具链。

2. **依赖树分层**：`langgraph-checkpoint` 是最底层的 LangGraph 包，不依赖核心框架；`langgraph` 核心包处于中间层，聚合了 checkpoint、prebuilt、sdk 三个依赖；CLI 在最上层。

3. **SDK 完全独立**：`langgraph-sdk` 仅依赖 `httpx` 和 `orjson`，可以在不安装 LangGraph 的环境中使用，实现了客户端与服务端的彻底解耦。

4. **命名空间共享**：`checkpoint`、`prebuilt`、核心包共享 `langgraph` Python 命名空间，对用户呈现为统一的包结构；CLI 和 SDK 使用独立命名空间。

5. **避免循环依赖**：`langgraph` 依赖 `prebuilt`，但 `prebuilt` 的发布依赖不包含 `langgraph`，通过精心设计打破了潜在的循环引用。

6. **可扩展 Checkpoint**：Checkpoint 接口下沉为独立包，使用宽松版本约束（`>=2.1.0,<5.0.0`），方便第三方实现自定义存储后端。
