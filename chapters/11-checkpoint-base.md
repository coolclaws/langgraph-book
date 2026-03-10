# 第 11 章 Checkpoint 抽象层

Checkpoint 是 LangGraph 持久化的基石。它让图的执行状态可以被保存、恢复、回溯，从而支持 human-in-the-loop、时间旅行调试、故障恢复等关键能力。本章深入分析 checkpoint 抽象层的源码设计，包括核心数据结构 `Checkpoint`、`CheckpointTuple`、`CheckpointMetadata`，以及基类 `BaseCheckpointSaver` 的完整接口定义。

---

## 11.1 Checkpoint 数据结构

`Checkpoint` 是一个 TypedDict，代表图在某个时间点的完整状态快照：

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/base/__init__.py

class Checkpoint(TypedDict):
    """State snapshot at a given point in time."""

    v: int
    """The version of the checkpoint format. Currently 1."""
    id: str
    """The ID of the checkpoint.
    This is both unique and monotonically increasing, so can be used for sorting
    checkpoints from first to last."""
    ts: str
    """The timestamp of the checkpoint in ISO 8601 format."""
    channel_values: dict[str, Any]
    """The values of the channels at the time of the checkpoint.
    Mapping from channel name to deserialized channel snapshot value."""
    channel_versions: ChannelVersions
    """The versions of the channels at the time of the checkpoint.
    The keys are channel names and the values are monotonically increasing
    version strings for each channel."""
    versions_seen: dict[str, ChannelVersions]
    """Map from node ID to map from channel name to version seen.
    This keeps track of the versions of the channels that each node has seen.
    Used to determine which nodes to execute next."""
    updated_channels: list[str] | None
    """The channels that were updated in this checkpoint."""
```

其中 `ChannelVersions` 的类型定义为：

```python
ChannelVersions = dict[str, str | int | float]
```

### 11.1.1 各字段详解

#### `v` — 格式版本号

当前为 `2`（由常量 `LATEST_VERSION = 2` 定义）。用于支持未来的格式迁移，checkpointer 在加载旧格式的 checkpoint 时可以据此进行兼容性处理。PostgresSaver 的代码中可以看到对 `v < 4` 的旧版本进行 pending_sends 迁移的逻辑。

#### `id` — Checkpoint ID

使用 UUID v6 生成：

```python
from langgraph.checkpoint.base.id import uuid6

# step 作为 clock_seq 参数
checkpoint_id = str(uuid6(clock_seq=step))
```

UUID v6 有两个重要特性：
- **唯一性**：每个 checkpoint 都有全局唯一的 ID
- **单调递增**：UUID v6 基于时间戳，天然递增，可用于排序。这就是为什么 SQL 查询中可以用 `ORDER BY checkpoint_id DESC` 来获取最新的 checkpoint

#### `ts` — 时间戳

ISO 8601 格式的 UTC 时间戳，如 `"2024-05-04T06:32:42.235444+00:00"`。

#### `channel_values` — Channel 值快照

这是最核心的字段，存储了图中所有 channel 在当前时刻的值。键是 channel 名称，值是该 channel 的快照。

```python
# 示例——一个聊天机器人图的 channel_values
{
    "messages": [HumanMessage("hello"), AIMessage("hi!")],
    "current_step": "analyze",
    "results": {"score": 0.95}
}
```

在 `create_checkpoint` 中，channel 值通过调用 `v.checkpoint()` 方法获取快照：

```python
for k, v in channels.items():
    if k not in checkpoint["channel_versions"]:
        continue
    try:
        values[k] = v.checkpoint()
    except EmptyChannelError:
        pass  # 空 channel 不保存
```

#### `channel_versions` — Channel 版本映射

记录每个 channel 的当前版本号。版本号是单调递增的（可以是 `int`、`float` 或 `str`），用于确定哪些 channel 在当前步骤中被更新了。

```python
# 示例
{
    "messages": 3,
    "current_step": 2,
    "results": 1
}
```

#### `versions_seen` — 节点已见版本映射

二层字典：外层键是节点名，内层是该节点上次执行时看到的各 channel 版本。这是 Pregel 引擎判断"哪些节点需要执行"的关键数据。

```python
# 示例
{
    "agent": {"messages": 2, "current_step": 1},
    "tool":  {"messages": 3, "results": 0}
}
```

Pregel 引擎的核心判断逻辑是：如果某个 channel 的当前版本（`channel_versions[ch]`）大于节点上次看到的版本（`versions_seen[node][ch]`），则该节点需要被重新执行。这实现了"数据驱动"的执行模型——只有当输入数据发生变化时，节点才会被触发。

#### `updated_channels` — 本次更新的 Channel 列表

可选字段，记录在本次 checkpoint 中哪些 channel 被更新了。可以为 `None`。

---

## 11.2 Checkpoint 的创建与复制

### 11.2.1 empty_checkpoint — 创建空白 Checkpoint

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/base/__init__.py

LATEST_VERSION = 2

def empty_checkpoint() -> Checkpoint:
    from datetime import datetime, timezone
    return Checkpoint(
        v=LATEST_VERSION,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
        pending_sends=[],
        updated_channels=None,
    )
```

注意 `clock_seq=-2`，这确保了初始 checkpoint 的 ID 在排序时位于所有正常步骤（step >= 0）之前。空 checkpoint 用于图的首次执行，此时还没有任何 channel 值。

### 11.2.2 create_checkpoint — 从当前状态创建新 Checkpoint

```python
def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, ChannelProtocol] | None,
    step: int,
    *,
    id: str | None = None,
) -> Checkpoint:
    """Create a checkpoint for the given channels."""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()
    if channels is None:
        values = checkpoint["channel_values"]
    else:
        values = {}
        for k, v in channels.items():
            if k not in checkpoint["channel_versions"]:
                continue
            try:
                values[k] = v.checkpoint()
            except EmptyChannelError:
                pass
    return Checkpoint(
        v=LATEST_VERSION,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
        pending_sends=checkpoint.get("pending_sends", []),
        updated_channels=None,
    )
```

关键设计点：
- 对每个 channel 调用 `v.checkpoint()` 获取其当前快照值
- 如果 channel 为空（抛出 `EmptyChannelError`），跳过该 channel
- `channel_versions` 和 `versions_seen` 从上一个 checkpoint 继承——这些版本信息在 Pregel 主循环中被单独更新
- `step` 作为 `uuid6` 的 `clock_seq` 参数，确保同一秒内创建的多个 checkpoint 也能正确排序

### 11.2.3 copy_checkpoint — 深拷贝 Checkpoint

```python
def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        id=checkpoint["id"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=checkpoint["channel_versions"].copy(),
        versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()},
        pending_sends=checkpoint.get("pending_sends", []).copy(),
        updated_channels=checkpoint.get("updated_channels", None),
    )
```

注意 `versions_seen` 需要二层拷贝（外层 dict 的 `.copy()` + 每个内层 dict 的 `.copy()`）。其他容器字段使用单层浅拷贝。这在 Pregel 引擎中很重要——引擎会修改 checkpoint 的版本信息，需要确保不影响原始 checkpoint。

---

## 11.3 CheckpointMetadata

`CheckpointMetadata` 记录 checkpoint 的元信息：

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/base/__init__.py

class CheckpointMetadata(TypedDict, total=False):
    """Metadata associated with a checkpoint."""

    source: Literal["input", "loop", "update", "fork"]
    """The source of the checkpoint."""
    step: int
    """The step number of the checkpoint."""
    parents: dict[str, str]
    """The IDs of the parent checkpoints. Mapping from checkpoint namespace to checkpoint ID."""
    run_id: str
    """The ID of the run that created this checkpoint."""
```

`total=False` 意味着所有字段都是可选的，这为向后兼容提供了灵活性。

### 11.3.1 source 字段详解

| 值 | 含义 | 何时产生 |
|---|------|---------|
| `"input"` | 从 invoke/stream/batch 的输入创建 | 图开始执行时的第一个 checkpoint |
| `"loop"` | 从 Pregel 主循环内部创建 | 每个 super-step 结束时 |
| `"update"` | 从手动状态更新创建 | 调用 `graph.update_state()` 时 |
| `"fork"` | 从另一个 checkpoint 复制创建 | 时间旅行/分支执行时 |

### 11.3.2 step 字段含义

- `-1`：第一个 "input" checkpoint（图开始执行前）
- `0`：第一个 "loop" checkpoint（第一个 super-step 完成后）
- `n`：第 n 步的 checkpoint

### 11.3.3 parents 字段

从 checkpoint namespace 到 checkpoint ID 的映射：

```python
# 示例——子图的 parents 字段
{
    "": "1ef4f797-8335-6428-8001-8a1503f9b875",           # 根图的 checkpoint ID
    "agent:abc123": "1ef4f797-8336-6428-8001-8a1503f9b876" # 上级子图的 checkpoint ID
}
```

这在子图场景下非常重要——子图的 checkpoint 需要知道它属于哪个父图的哪个步骤，以便在恢复时正确建立父子关系。

### 11.3.4 Metadata 处理函数

```python
def get_checkpoint_metadata(
    config: RunnableConfig, metadata: CheckpointMetadata
) -> CheckpointMetadata:
    metadata = {
        k: v.replace("\u0000", "") if isinstance(v, str) else v
        for k, v in metadata.items()
    }
    for obj in (config.get("metadata"), config.get("configurable")):
        if not obj:
            continue
        for key, v in obj.items():
            if key in metadata or key in EXCLUDED_METADATA_KEYS or key.startswith("__"):
                continue
            elif isinstance(v, str):
                metadata[key] = v.replace("\u0000", "")
            elif isinstance(v, (int, bool, float)):
                metadata[key] = v
    return metadata
```

这个函数做了几件事：
1. **去除 null 字节**（`\u0000`）：某些数据库（如 PostgreSQL 的 JSONB 列）不支持 null 字节
2. **合并 config 中的额外元数据**：从 `config["metadata"]` 和 `config["configurable"]` 中提取
3. **过滤系统键**：排除以 `__` 开头的内部键和 `EXCLUDED_METADATA_KEYS` 中的键
4. **只保留原始类型**：只接受 `str`、`int`、`bool`、`float`，不存储复杂对象

被排除的系统键：

```python
EXCLUDED_METADATA_KEYS = {
    "thread_id",
    "checkpoint_id",
    "checkpoint_ns",
    "checkpoint_map",
    "langgraph_step",
    "langgraph_node",
    "langgraph_triggers",
    "langgraph_path",
    "langgraph_checkpoint_ns",
}
```

还有一个变体函数 `get_serializable_checkpoint_metadata`，它在 `get_checkpoint_metadata` 的基础上额外移除了 `writes` 字段（因为 writes 可能包含不适合存储在 metadata 中的大型对象）：

```python
def get_serializable_checkpoint_metadata(
    config: RunnableConfig, metadata: CheckpointMetadata
) -> CheckpointMetadata:
    checkpoint_metadata = get_checkpoint_metadata(config, metadata)
    if "writes" in checkpoint_metadata:
        checkpoint_metadata.pop("writes")
    return checkpoint_metadata
```

---

## 11.4 CheckpointTuple

`CheckpointTuple` 将 checkpoint 与其关联的配置和元数据打包在一起：

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/base/__init__.py

class CheckpointTuple(NamedTuple):
    """A tuple containing a checkpoint and its associated data."""

    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None
    pending_writes: list[PendingWrite] | None = None
```

各字段含义：

| 字段 | 类型 | 说明 |
|------|------|------|
| `config` | `RunnableConfig` | 包含 `thread_id`、`checkpoint_ns`、`checkpoint_id` |
| `checkpoint` | `Checkpoint` | 实际的状态快照 |
| `metadata` | `CheckpointMetadata` | 元数据（source、step 等） |
| `parent_config` | `RunnableConfig \| None` | 父 checkpoint 的配置，用于回溯历史 |
| `pending_writes` | `list[PendingWrite] \| None` | 尚未完成的写入操作 |

### 11.4.1 PendingWrite

```python
PendingWrite = tuple[str, str, Any]
```

PendingWrite 是一个三元组 `(task_id, channel, value)`，代表某个 task 对某个 channel 的写入操作。这些写入可能在 task 完成前就被保存到存储中（通过 `put_writes`），用于崩溃恢复——即使进程在执行过程中意外终止，已保存的 pending writes 也不会丢失。

当图从 checkpoint 恢复时，Pregel 引擎会加载 pending_writes，对于已完成的 task 直接使用保存的结果，无需重新执行。

---

## 11.5 BaseCheckpointSaver 接口

`BaseCheckpointSaver` 是所有 checkpointer 实现的基类：

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/base/__init__.py

class BaseCheckpointSaver(Generic[V]):
    """Base class for creating a graph checkpointer."""

    serde: SerializerProtocol = JsonPlusSerializer()

    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        self.serde = maybe_add_typed_methods(serde or self.serde)
```

泛型参数 `V` 代表版本号类型，可以是 `int`、`float` 或 `str`。不同的 checkpointer 实现使用不同的版本号类型——InMemorySaver 用 `int`，SqliteSaver 和 PostgresSaver 用 `str`。

### 11.5.1 读取方法

#### get — 获取 Checkpoint（便捷方法）

```python
def get(self, config: RunnableConfig) -> Checkpoint | None:
    if value := self.get_tuple(config):
        return value.checkpoint
```

简单地委托给 `get_tuple`，只返回 Checkpoint 本身。

#### get_tuple — 获取完整的 CheckpointTuple（需子类实现）

```python
def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
    raise NotImplementedError
```

这是子类必须实现的核心方法，返回包含 checkpoint、metadata、parent_config 和 pending_writes 的完整信息。

#### list — 列出 Checkpoints

```python
def list(
    self,
    config: RunnableConfig | None,
    *,
    filter: dict[str, Any] | None = None,
    before: RunnableConfig | None = None,
    limit: int | None = None,
) -> Iterator[CheckpointTuple]:
    raise NotImplementedError
```

支持三种过滤方式：
- `config`：按 `thread_id` 和 `checkpoint_ns` 过滤
- `filter`：按 metadata 字段过滤（如 `{"source": "loop"}`）
- `before`：只返回指定 checkpoint 之前的记录（用于时间旅行）

### 11.5.2 写入方法

#### put — 存储 Checkpoint

```python
def put(
    self,
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    new_versions: ChannelVersions,
) -> RunnableConfig:
    raise NotImplementedError
```

存储一个完整的 checkpoint，返回更新后的 `RunnableConfig`（包含新生成的 `checkpoint_id`）。`new_versions` 参数告诉 checkpointer 哪些 channel 在本次写入中有新的版本——某些实现（如 PostgresSaver）会用这个信息做增量存储。

#### put_writes — 存储中间写入

```python
def put_writes(
    self,
    config: RunnableConfig,
    writes: Sequence[tuple[str, Any]],
    task_id: str,
    task_path: str = "",
) -> None:
    raise NotImplementedError
```

在 task 执行过程中保存中间写入结果。`writes` 是 `(channel, value)` 对的列表。这些写入与特定的 checkpoint 和 task 关联。

### 11.5.3 删除方法

```python
def delete_thread(self, thread_id: str) -> None:
    """Delete all checkpoints and writes associated with a specific thread ID."""
    raise NotImplementedError

def delete_for_runs(self, run_ids: Sequence[str]) -> None:
    """Delete all checkpoints and writes associated with the given run IDs."""
    raise NotImplementedError
```

### 11.5.4 管理方法

```python
def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
    """Copy all checkpoints and writes from one thread to another."""
    raise NotImplementedError

def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
    """Prune checkpoints for the given threads.
    strategy: "keep_latest" retains only the most recent checkpoint per namespace.
              "delete" removes all checkpoints."""
    raise NotImplementedError
```

`copy_thread` 用于分支执行场景——将一个 thread 的完整历史复制到新的 thread ID 下，然后在新 thread 上继续执行。

`prune` 用于清理存储空间，`"keep_latest"` 策略只保留每个 namespace 的最新 checkpoint。

### 11.5.5 异步方法

每个同步方法都有对应的异步版本（以 `a` 前缀命名）：

```python
async def aget(self, config) -> Checkpoint | None: ...
async def aget_tuple(self, config) -> CheckpointTuple | None: ...
async def alist(self, config, *, filter, before, limit) -> AsyncIterator[CheckpointTuple]: ...
async def aput(self, config, checkpoint, metadata, new_versions) -> RunnableConfig: ...
async def aput_writes(self, config, writes, task_id, task_path) -> None: ...
async def adelete_thread(self, thread_id) -> None: ...
async def adelete_for_runs(self, run_ids) -> None: ...
async def acopy_thread(self, source_thread_id, target_thread_id) -> None: ...
async def aprune(self, thread_ids, *, strategy) -> None: ...
```

注意 `alist` 返回的是 `AsyncIterator`——它使用 `yield` 在异步上下文中逐个产出结果。

### 11.5.6 版本号生成

```python
def get_next_version(self, current: V | None, channel: None) -> V:
    """Generate the next version ID for a channel.
    Default is to use integer versions, incrementing by 1."""
    if isinstance(current, str):
        raise NotImplementedError
    elif current is None:
        return 1
    else:
        return current + 1
```

默认实现使用整数版本号，每次递增 1。`SqliteSaver` 和 `PostgresSaver` 重写了此方法，使用带随机后缀的字符串版本号（如 `"00000003.0.7234561"`）。随机后缀的作用是在分布式场景下避免版本冲突——两个并发写入即使在同一微秒发生，也不太可能生成相同的版本号。

---

## 11.6 thread_id + checkpoint_ns 隔离

LangGraph 的 checkpoint 使用**两级隔离键**：

### 11.6.1 thread_id — 线程级隔离

`thread_id` 是最基本的隔离单元。不同的 `thread_id` 之间的 checkpoint 完全独立：

```python
# 两个不同的 thread，状态互不影响
config_1 = {"configurable": {"thread_id": "user-alice"}}
config_2 = {"configurable": {"thread_id": "user-bob"}}

graph.invoke({"messages": ["hello"]}, config_1)  # Alice 的对话
graph.invoke({"messages": ["hi"]}, config_2)     # Bob 的对话
```

### 11.6.2 checkpoint_ns — 子图级隔离

`checkpoint_ns`（checkpoint namespace）用于隔离主图和子图的 checkpoint。格式为用 `|` 分隔的层级路径：

```
""                                    ← 根图（默认值）
"agent:abc123"                        ← 名为 "agent" 的子图，task ID 为 abc123
"agent:abc123|tool:def456"            ← agent 子图中的 tool 子图
"agent:abc123|1|tool:def456"          ← 带有数字消歧的命名空间
```

数字消歧符（如 `|1|`）用于并发执行时区分同一节点的不同 task 实例。

在 `config` 中，这些值位于 `configurable` 字典中：

```python
config = {
    "configurable": {
        "thread_id": "my-thread",
        "checkpoint_ns": "",           # 根图
        "checkpoint_id": "1ef4f797...", # 可选，指定具体的 checkpoint
    }
}
```

### 11.6.3 隔离机制的实际作用

这种两级隔离设计解决了几个关键问题：

1. **多用户并发**：每个用户使用不同的 `thread_id`，状态天然隔离
2. **子图独立性**：子图有自己的 `checkpoint_ns`，其内部状态不会与父图或其他子图混淆
3. **时间旅行**：在同一个 `(thread_id, checkpoint_ns)` 内，可以通过 `checkpoint_id` 回溯到任意历史时刻
4. **分支执行**：通过 `copy_thread` 将一个 thread 复制到新的 `thread_id`，然后独立继续执行

---

## 11.7 WRITES_IDX_MAP：特殊写入类型

LangGraph 定义了几种特殊的写入类型，使用负数索引与普通写入区分：

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/base/__init__.py

WRITES_IDX_MAP = {ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}
```

| 类型 | 索引 | 说明 |
|------|------|------|
| `ERROR` | -1 | 节点执行出错时的错误信息 |
| `SCHEDULED` | -2 | 调度信息 |
| `INTERRUPT` | -3 | 中断信息（interrupt() 函数产生的 Interrupt 对象） |
| `RESUME` | -4 | 恢复执行时的值（Command(resume=...) 提供的值） |

这些特殊常量定义在 `langgraph/checkpoint/serde/types.py` 中。

在 checkpointer 的 `put_writes` 实现中，写入类型会影响 SQL 的冲突处理策略：

- **特殊写入**使用 `INSERT OR REPLACE`（可以覆盖旧值）——例如，如果一个 task 先出错后被重试成功，ERROR 记录应该被覆盖
- **普通写入**使用 `INSERT OR IGNORE`（幂等性）——相同的写入不会被重复保存

---

## 11.8 Serializer 配置

`BaseCheckpointSaver` 通过 `serde` 属性持有一个序列化器：

```python
class BaseCheckpointSaver(Generic[V]):
    serde: SerializerProtocol = JsonPlusSerializer()

    def __init__(self, *, serde: SerializerProtocol | None = None) -> None:
        self.serde = maybe_add_typed_methods(serde or self.serde)
```

`maybe_add_typed_methods` 是一个兼容适配器，确保旧式序列化器（只有 `dumps`/`loads` 方法）也能正常工作：

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/serde/base.py

def maybe_add_typed_methods(
    serde: SerializerProtocol | UntypedSerializerProtocol,
) -> SerializerProtocol:
    if not isinstance(serde, SerializerProtocol):
        return SerializerCompat(serde)
    return serde
```

### 11.8.1 with_allowlist

`BaseCheckpointSaver` 提供了 `with_allowlist` 方法，用于扩展 msgpack 的反序列化白名单：

```python
def with_allowlist(
    self, extra_allowlist: Collection[tuple[str, ...]]
) -> BaseCheckpointSaver[V]:
    """Return a shallow clone with a derived msgpack allowlist."""
    serde = _with_msgpack_allowlist(self.serde, extra_allowlist)
    if serde is self.serde:
        return self
    clone = copy.copy(self)
    clone.serde = maybe_add_typed_methods(serde)
    return clone
```

当图的 State 中包含自定义类型时，需要通过 allowlist 显式允许反序列化：

```python
saver = SqliteSaver(conn)
saver = saver.with_allowlist([("myapp.models", "MyCustomModel")])
```

`_with_msgpack_allowlist` 会递归处理嵌套的序列化器（例如 `EncryptedSerializer` 包裹的 `JsonPlusSerializer`）。

---

## 11.9 Checkpoint 在 Pregel 主循环中的角色

虽然本章聚焦于抽象层，但了解 checkpoint 在 Pregel 主循环中的角色有助于理解设计意图：

```
┌─────────────────────────────────────────────────┐
│              Pregel 主循环                        │
│                                                 │
│  1. 加载 checkpoint: get_tuple(config)           │
│  2. 恢复 channel 状态（从 channel_values）        │
│  3. 确定待执行节点                                │
│     比较 channel_versions vs versions_seen       │
│  4. 并行执行节点                                  │
│     ├─ 执行中: put_writes(pending writes)        │
│     └─ 完成: 更新 channel                        │
│  5. 创建新 checkpoint: create_checkpoint + put   │
│  6. 检查是否有待执行节点                           │
│     ├─ 有: 回到步骤 3（新的 super-step）           │
│     └─ 无: 结束                                  │
└─────────────────────────────────────────────────┘
```

每个 super-step 结束后，Pregel 引擎调用 `put` 保存一个新的 checkpoint。这确保了即使在步骤间崩溃，也能从最后一个 checkpoint 恢复。

---

## 11.10 config_specs 属性

```python
@property
def config_specs(self) -> list:
    """Define the configuration options for the checkpoint saver."""
    return []
```

`config_specs` 属性允许 checkpointer 声明它需要哪些配置字段。在 LangGraph Platform 的 HTTP 部署中，这些字段会被暴露为 API 参数。默认返回空列表。

---

## 11.11 设计考量

### 11.11.1 为什么用 TypedDict 而非 dataclass

`Checkpoint` 和 `CheckpointMetadata` 使用 `TypedDict` 而非 `dataclass`，原因是：

1. **JSON 兼容**：TypedDict 本质上是 dict，可以直接传递给 `json.dumps` 或 `ormsgpack.packb`
2. **向后兼容**：添加新字段时不会破坏已有的序列化数据（旧数据中缺少的字段在反序列化时自然不存在）
3. **灵活性**：`total=False` 让所有字段可选，适应不同版本的 checkpoint 格式

### 11.11.2 为什么 CheckpointTuple 用 NamedTuple

`CheckpointTuple` 使用 `NamedTuple`，因为它是一个不可变的数据传输对象。NamedTuple 比 dataclass 更轻量，支持解包（`config, checkpoint, metadata, *_ = get_tuple(config)`），且天然不可变——这与 checkpoint 的语义一致（已保存的 checkpoint 不应被修改）。

### 11.11.3 pending_sends 的作用

`Checkpoint` 中有一个 `pending_sends` 字段（在 `copy_checkpoint` 和 `create_checkpoint` 中可见），它记录了通过 `Send` 对象发送给特定节点的消息。这些消息在下一个 super-step 中被消费，用于实现 map-reduce 等并行模式。

---

## 本章要点

1. **Checkpoint** 是一个 TypedDict，核心字段包括 `id`（UUID v6，单调递增）、`channel_values`（所有 channel 的当前快照值）、`channel_versions`（版本号映射）、`versions_seen`（每个节点看到的版本号映射）

2. **CheckpointMetadata** 记录 checkpoint 的来源（`source`: input/loop/update/fork）、步骤号（`step`: -1 起始）、父 checkpoint 关系（`parents`）和运行 ID（`run_id`）

3. **CheckpointTuple** 将 checkpoint、config、metadata、parent_config 和 pending_writes 打包在一起，是 checkpointer 与 Pregel 引擎之间的数据传输对象

4. **PendingWrite** 是 `(task_id, channel, value)` 三元组，用于在 task 执行过程中保存中间写入，支持崩溃恢复

5. **BaseCheckpointSaver** 定义了 `get_tuple`/`list`/`put`/`put_writes`/`delete_thread` 等核心接口，每个方法都有同步和异步版本

6. **两级隔离**：`thread_id` 隔离不同的对话/工作流实例，`checkpoint_ns` 隔离主图与子图的状态。格式为 `"node_name:task_id|..."` 的层级路径

7. **WRITES_IDX_MAP** 定义了四种特殊写入类型（ERROR=-1, SCHEDULED=-2, INTERRUPT=-3, RESUME=-4），使用负数索引与普通写入区分，并影响 SQL 的冲突处理策略

8. **版本号生成**：默认使用整数递增。`get_next_version` 方法可被子类重写——SqliteSaver 和 PostgresSaver 使用带随机后缀的字符串版本号以支持并发场景
