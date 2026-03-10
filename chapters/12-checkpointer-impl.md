# 第 12 章 三种 Checkpointer 实现 + Serde

上一章介绍了 `BaseCheckpointSaver` 的抽象接口，本章将深入分析三种官方 checkpointer 实现——InMemorySaver（内存）、SqliteSaver（SQLite）、PostgresSaver（PostgreSQL），以及支撑序列化/反序列化的 Serde 子系统。

---

## 12.1 InMemorySaver：内存实现

InMemorySaver 将所有 checkpoint 数据存储在 Python 字典中，是最简单的 checkpointer 实现。它不需要外部依赖，非常适合开发、测试和演示。

### 12.1.1 使用方式

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 12.1.2 基本特点

- **零依赖**：不需要安装额外的数据库驱动
- **内存存储**：所有数据存储在 Python 字典中，进程退出后丢失
- **完全同步和异步**：同时实现了同步和异步接口
- **线程安全**：通过 Python GIL 和内部数据结构保证基本的线程安全
- **不支持跨进程共享**：数据只在当前进程内可见

### 12.1.3 内部数据结构

InMemorySaver 的存储结构通常是嵌套字典，以 `(thread_id, checkpoint_ns, checkpoint_id)` 三元组为键：

```
storage = {
    (thread_id, checkpoint_ns): OrderedDict({
        checkpoint_id_1: {
            "checkpoint": Checkpoint,
            "metadata": CheckpointMetadata,
            "parent_checkpoint_id": str | None,
        },
        checkpoint_id_2: { ... },
    }),
}

writes_storage = {
    (thread_id, checkpoint_ns, checkpoint_id): {
        (task_id, idx): (channel, type, value),
    },
}
```

由于所有数据都在内存中，`get_tuple`、`list`、`put` 等方法就是简单的字典操作，没有 SQL 查询和序列化/反序列化的开销。

### 12.1.4 适用场景

- 单元测试和集成测试
- Jupyter Notebook 交互式开发
- 简单的单进程应用原型
- 学习和理解 checkpoint 机制

---

## 12.2 SqliteSaver：SQLite 实现

`SqliteSaver` 将 checkpoint 存储在 SQLite 数据库中，适合单机轻量级部署：

```python
# 源码路径: libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/__init__.py

class SqliteSaver(BaseCheckpointSaver[str]):
    """A checkpoint saver that stores checkpoints in a SQLite database."""

    conn: sqlite3.Connection
    is_setup: bool

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.jsonplus_serde = JsonPlusSerializer()
        self.conn = conn
        self.is_setup = False
        self.lock = threading.Lock()
```

注意泛型参数为 `str`——SqliteSaver 使用字符串版本号。

### 12.2.1 数据库表结构

`SqliteSaver` 在 `setup()` 方法中创建两张表：

```sql
-- 启用 WAL 模式提升并发读性能
PRAGMA journal_mode=WAL;

-- Checkpoint 主表
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- 中间写入表
CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
```

几个设计要点：

1. **三元组主键**：`(thread_id, checkpoint_ns, checkpoint_id)` 唯一标识一个 checkpoint
2. **WAL 模式**：Write-Ahead Logging 允许读写并发，提升性能
3. **BLOB 存储**：checkpoint 和 writes 的值以二进制 BLOB 存储，`type` 字段标识序列化格式（如 `"msgpack"`、`"json"` 等）
4. **writes 表**：独立存储中间写入，通过五元组 `(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)` 唯一标识

### 12.2.2 线程安全机制

```python
self.lock = threading.Lock()

@contextmanager
def cursor(self, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
    """Get a cursor for the SQLite database."""
    with self.lock:
        self.setup()
        cur = self.conn.cursor()
        try:
            yield cur
        finally:
            if transaction:
                self.conn.commit()
            cur.close()
```

所有数据库操作都在持有 `threading.Lock()` 的情况下执行。`setup()` 在第一次获取 cursor 时调用（lazy initialization），之后通过 `is_setup` 标志跳过。

### 12.2.3 get_tuple 实现

```python
def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
    with self.cursor(transaction=False) as cur:
        # 如果指定了 checkpoint_id，精确查询
        if checkpoint_id := get_checkpoint_id(config):
            cur.execute(
                "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, "
                "checkpoint, metadata FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                (str(config["configurable"]["thread_id"]), checkpoint_ns, checkpoint_id),
            )
        else:
            # 否则取最新的 checkpoint
            cur.execute(
                "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, "
                "checkpoint, metadata FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = ? "
                "ORDER BY checkpoint_id DESC LIMIT 1",
                (str(config["configurable"]["thread_id"]), checkpoint_ns),
            )
        if value := cur.fetchone():
            (thread_id, checkpoint_id, parent_checkpoint_id,
             type, checkpoint, metadata) = value
            # 查询关联的 pending writes
            cur.execute(
                "SELECT task_id, channel, type, value FROM writes "
                "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
                "ORDER BY task_id, idx",
                (thread_id, checkpoint_ns, checkpoint_id),
            )
            return CheckpointTuple(
                config,
                self.serde.loads_typed((type, checkpoint)),
                json.loads(metadata) if metadata is not None else {},
                parent_config_or_none,
                [(task_id, channel, self.serde.loads_typed((type, value)))
                 for task_id, channel, type, value in cur],
            )
```

注意 `ORDER BY checkpoint_id DESC LIMIT 1`——因为 checkpoint_id 是 UUID v6（单调递增），所以降序排列的第一个就是最新的 checkpoint。

### 12.2.4 put 实现

```python
def put(self, config, checkpoint, metadata, new_versions) -> RunnableConfig:
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
    type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
    serialized_metadata = json.dumps(
        get_checkpoint_metadata(config, metadata), ensure_ascii=False
    ).encode("utf-8", "ignore")
    with self.cursor() as cur:
        cur.execute(
            "INSERT OR REPLACE INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
            "type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(thread_id), checkpoint_ns, checkpoint["id"],
             config["configurable"].get("checkpoint_id"),
             type_, serialized_checkpoint, serialized_metadata),
        )
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint["id"],
        }
    }
```

要点：
- checkpoint 使用 `serde.dumps_typed` 序列化，返回 `(type, bytes)` 二元组
- metadata 使用 `json.dumps` 单独序列化——这样做的好处是 metadata 可以被 SQL 查询过滤
- `parent_checkpoint_id` 取自当前 config 中的 `checkpoint_id`（即上一个 checkpoint 的 ID）
- 使用 `INSERT OR REPLACE` 实现 upsert 语义

### 12.2.5 put_writes 实现

```python
def put_writes(self, config, writes, task_id, task_path="") -> None:
    query = (
        "INSERT OR REPLACE INTO writes ..."
        if all(w[0] in WRITES_IDX_MAP for w in writes)
        else "INSERT OR IGNORE INTO writes ..."
    )
    with self.cursor() as cur:
        cur.executemany(
            query,
            [
                (str(config["configurable"]["thread_id"]),
                 str(config["configurable"]["checkpoint_ns"]),
                 str(config["configurable"]["checkpoint_id"]),
                 task_id,
                 WRITES_IDX_MAP.get(channel, idx),
                 channel,
                 *self.serde.dumps_typed(value))
                for idx, (channel, value) in enumerate(writes)
            ],
        )
```

关键设计：
- **特殊写入**（全部 channel 都在 `WRITES_IDX_MAP` 中）使用 `INSERT OR REPLACE`：可以覆盖旧值
- **普通写入**（至少有一个 channel 不在 `WRITES_IDX_MAP` 中）使用 `INSERT OR IGNORE`：保证幂等性
- **idx 映射**：特殊通道使用负数索引（ERROR=-1, INTERRUPT=-3 等），普通通道使用从 0 开始的递增索引

### 12.2.6 版本号生成

```python
def get_next_version(self, current: str | None, channel: None) -> str:
    if current is None:
        current_v = 0
    elif isinstance(current, int):
        current_v = current
    else:
        current_v = int(current.split(".")[0])
    next_v = current_v + 1
    next_h = random.random()
    return f"{next_v:032}.{next_h:016}"
```

SQLite 版本号格式：`"00000000000000000000000000000003.0.723456123456"` — 包含零填充的递增计数器和随机小数。这种格式既保证了排序正确性（字符串排序等价于数值排序），又通过随机后缀避免了并发写入时的冲突。

### 12.2.7 工厂方法

```python
@classmethod
@contextmanager
def from_conn_string(cls, conn_string: str) -> Iterator[SqliteSaver]:
    with closing(
        sqlite3.connect(conn_string, check_same_thread=False)
    ) as conn:
        yield cls(conn)
```

使用上下文管理器自动管理连接生命周期：

```python
# 内存数据库（测试用）
with SqliteSaver.from_conn_string(":memory:") as saver:
    graph = builder.compile(checkpointer=saver)

# 文件数据库（持久化）
with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
    graph = builder.compile(checkpointer=saver)
```

### 12.2.8 异步支持

SqliteSaver 的所有异步方法抛出 `NotImplementedError`：

```python
_AIO_ERROR_MSG = (
    "The SqliteSaver does not support async methods. "
    "Consider using AsyncSqliteSaver instead.\n"
    "from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver"
)

async def aget_tuple(self, config) -> CheckpointTuple | None:
    raise NotImplementedError(_AIO_ERROR_MSG)
```

需要异步支持时，应使用 `AsyncSqliteSaver`（位于 `langgraph.checkpoint.sqlite.aio`），它基于 `aiosqlite` 库。

---

## 12.3 PostgresSaver：PostgreSQL 实现

`PostgresSaver` 将 checkpoint 存储在 PostgreSQL 数据库中，适合生产环境：

```python
# 源码路径: libs/checkpoint-postgres/langgraph/checkpoint/postgres/__init__.py

class PostgresSaver(BasePostgresSaver):
    """Checkpointer that stores checkpoints in a Postgres database."""

    lock: threading.Lock

    def __init__(
        self,
        conn: _internal.Conn,
        pipe: Pipeline | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if isinstance(conn, ConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single Connection, not ConnectionPool."
            )
        self.conn = conn
        self.pipe = pipe
        self.lock = threading.Lock()
        self.supports_pipeline = Capabilities().has_pipeline()
```

PostgresSaver 继承自 `BasePostgresSaver`，后者定义了 SQL 语句和通用辅助方法。

### 12.3.1 数据库表结构

PostgresSaver 使用迁移系统管理表结构。核心表有四张：

```sql
-- 迁移版本跟踪表
CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);

-- Checkpoint 主表——注意使用 JSONB 类型
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Channel 值 BLOB 存储——拆分出来的大型 channel 值
CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

-- 中间写入表
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
```

后续迁移还添加了索引：

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
```

### 12.3.2 与 SQLite 的架构差异

PostgresSaver 相比 SqliteSaver 有几个显著的架构差异：

#### 差异一：JSONB vs BLOB

```
SQLite:    checkpoint 整体存为 BLOB（二进制序列化）
PostgreSQL: checkpoint 存为 JSONB，复杂 channel_values 拆分到 checkpoint_blobs 表
```

在 `put` 方法中，原始类型留在 checkpoint JSONB 中，复杂类型单独存储：

```python
def put(self, config, checkpoint, metadata, new_versions) -> RunnableConfig:
    copy = checkpoint.copy()
    copy["channel_values"] = copy["channel_values"].copy()

    # 分离原始值和复杂值
    blob_values = {}
    for k, v in checkpoint["channel_values"].items():
        if v is None or isinstance(v, (str, int, float, bool)):
            pass  # 原始值留在 checkpoint JSON 中
        else:
            blob_values[k] = copy["channel_values"].pop(k)

    with self._cursor(pipeline=True) as cur:
        # 复杂值写入 blobs 表
        if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
            cur.executemany(
                self.UPSERT_CHECKPOINT_BLOBS_SQL,
                self._dump_blobs(thread_id, checkpoint_ns, blob_values, blob_versions),
            )
        # checkpoint JSON + metadata JSON 写入 checkpoints 表
        cur.execute(
            self.UPSERT_CHECKPOINTS_SQL,
            (thread_id, checkpoint_ns, checkpoint["id"], checkpoint_id,
             Jsonb(copy),
             Jsonb(get_serializable_checkpoint_metadata(config, metadata))),
        )
```

这种设计的优势：
- **查询效率**：metadata 存为 JSONB，可以用 PostgreSQL 的 JSONB 运算符直接过滤
- **存储效率**：blobs 表按 `(channel, version)` 存储，相同版本的 blob 只存一份
- **增量更新**：只有 `new_versions` 中的 channel 需要写入新的 blob

#### 差异二：Pipeline 模式

PostgresSaver 支持 psycopg 的 pipeline 模式，将多个 SQL 命令打包发送以减少网络往返：

```python
@contextmanager
def _cursor(self, *, pipeline: bool = False) -> Iterator[Cursor[DictRow]]:
    with self.lock, _internal.get_connection(self.conn) as conn:
        if self.pipe:
            # 全局 pipeline 模式
            with conn.cursor(binary=True, row_factory=dict_row) as cur:
                yield cur
        elif pipeline:
            if self.supports_pipeline:
                # 按需 pipeline 模式
                with conn.pipeline(), conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur
            else:
                # 回退到事务模式
                with conn.transaction(), conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur
        else:
            with conn.cursor(binary=True, row_factory=dict_row) as cur:
                yield cur
```

#### 差异三：迁移系统

PostgresSaver 使用有序的迁移列表管理数据库 schema 变更：

```python
def setup(self) -> None:
    with self._cursor() as cur:
        cur.execute(self.MIGRATIONS[0])  # 创建 checkpoint_migrations 表
        results = cur.execute(
            "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
        )
        row = results.fetchone()
        version = row["v"] if row else -1
        for v, migration in zip(
            range(version + 1, len(self.MIGRATIONS)),
            self.MIGRATIONS[version + 1:],
        ):
            cur.execute(migration)
            cur.execute("INSERT INTO checkpoint_migrations (v) VALUES (%s)", (v,))
```

每次 `setup()` 调用时依次检查并应用所有未执行的迁移。

#### 差异四：连接池支持

```python
# 单连接模式
with PostgresSaver.from_conn_string(DB_URI) as saver:
    graph = builder.compile(checkpointer=saver)

# Pipeline 模式
with PostgresSaver.from_conn_string(DB_URI, pipeline=True) as saver:
    graph = builder.compile(checkpointer=saver)

# 连接池模式（生产推荐）
from psycopg_pool import ConnectionPool
pool = ConnectionPool(DB_URI)
saver = PostgresSaver(pool)
saver.setup()
```

### 12.3.3 ShallowPostgresSaver

PostgresSaver 包中还提供了 `ShallowPostgresSaver` 变体。它只保留每个 `(thread_id, checkpoint_ns)` 的最新 checkpoint，不维护完整历史链。适用于不需要时间旅行功能且希望减少存储开销的场景。

### 12.3.4 delete_thread 实现

```python
def delete_thread(self, thread_id: str) -> None:
    with self._cursor(pipeline=True) as cur:
        cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (str(thread_id),))
        cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (str(thread_id),))
        cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (str(thread_id),))
```

注意需要同时删除三张表中的数据。使用 pipeline 模式将三条 DELETE 语句打包发送。

---

## 12.4 三种实现的对比

| 特性 | InMemorySaver | SqliteSaver | PostgresSaver |
|------|---------------|-------------|---------------|
| 持久化 | 否（内存） | 是（文件） | 是（数据库） |
| 异步支持 | 是 | 需 AsyncSqliteSaver | 需 AsyncPostgresSaver |
| 线程安全 | GIL | threading.Lock | threading.Lock + 连接池 |
| 版本号类型 | `int` | `str` | `str` |
| 适用场景 | 开发/测试 | 单机轻量部署 | 生产环境 |
| 外部依赖 | 无 | sqlite3（标准库） | psycopg, psycopg_pool |
| 连接池 | N/A | N/A | 支持 |
| Pipeline 模式 | N/A | N/A | 支持 |
| 迁移系统 | N/A | 自动建表 | 有序迁移 |
| 存储格式 | Python 对象 | BLOB + JSON metadata | JSONB + BYTEA |
| channel_values | 直接存储 | 整体序列化 | 原始值 JSONB + 复杂值 BLOB |

---

## 12.5 Serde 子系统

LangGraph 的序列化/反序列化（Serde）子系统位于 `langgraph/checkpoint/serde/` 目录下：

```
libs/checkpoint/langgraph/checkpoint/serde/
├── __init__.py
├── base.py           # SerializerProtocol, CipherProtocol
├── jsonplus.py       # JsonPlusSerializer（默认序列化器）
├── encrypted.py      # EncryptedSerializer（加密装饰器）
├── _msgpack.py       # msgpack 安全类型白名单
├── event_hooks.py    # 序列化事件钩子（审计用）
└── types.py          # 特殊通道常量（ERROR, INTERRUPT 等）
```

### 12.5.1 SerializerProtocol

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/serde/base.py

@runtime_checkable
class SerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...
    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...
```

`dumps_typed` 返回 `(类型字符串, 字节数据)` 二元组。常见的类型字符串包括：
- `"msgpack"` — ormsgpack 二进制格式
- `"json"` — JSON 文本格式
- `"bytes"` — 原始字节
- `"null"` — None 值
- `"pickle"` — pickle 格式（需启用 fallback）
- `"msgpack+aes"` — 加密后的 msgpack

还有一个兼容适配器 `SerializerCompat`，包装只有 `dumps`/`loads` 方法的旧式序列化器：

```python
class SerializerCompat(SerializerProtocol):
    def __init__(self, serde: UntypedSerializerProtocol) -> None:
        self.serde = serde

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return type(obj).__name__, self.serde.dumps(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return self.serde.loads(data[1])
```

### 12.5.2 JsonPlusSerializer

`JsonPlusSerializer` 是默认的序列化器，名字中虽有 "Json"，但核心使用的是 `ormsgpack`（高性能 MessagePack 库）：

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py

class JsonPlusSerializer(SerializerProtocol):
    """Serializer that uses ormsgpack, with optional fallbacks.

    Security note: This serializer is intended for use within the
    BaseCheckpointSaver class. It should not be used on untrusted python objects.
    """

    def __init__(
        self,
        *,
        pickle_fallback: bool = False,
        allowed_json_modules: Iterable[tuple[str, ...]] | Literal[True] | None = None,
        allowed_msgpack_modules: AllowedMsgpackModules | Literal[True] | None = ...,
        __unpack_ext_hook__: Callable[[int, bytes], Any] | None = None,
    ) -> None:
        # ...
```

#### dumps_typed 流程

```python
def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
    if obj is None:
        return "null", EMPTY_BYTES
    elif isinstance(obj, bytes):
        return "bytes", obj
    elif isinstance(obj, bytearray):
        return "bytearray", obj
    else:
        try:
            return "msgpack", _msgpack_enc(obj)
        except ormsgpack.MsgpackEncodeError as exc:
            if self.pickle_fallback:
                return "pickle", pickle.dumps(obj)
            raise exc
```

优先级：`None` -> `bytes` -> `bytearray` -> msgpack -> (fallback) pickle

#### loads_typed 流程

```python
def loads_typed(self, data: tuple[str, bytes]) -> Any:
    type_, data_ = data
    if type_ == "null":
        return None
    elif type_ == "bytes":
        return data_
    elif type_ == "bytearray":
        return bytearray(data_)
    elif type_ == "json":
        return json.loads(data_, object_hook=self._reviver)
    elif type_ == "msgpack":
        return ormsgpack.unpackb(
            data_, ext_hook=self._unpack_ext_hook,
            option=ormsgpack.OPT_NON_STR_KEYS
        )
    elif self.pickle_fallback and type_ == "pickle":
        return pickle.loads(data_)
    else:
        raise NotImplementedError(f"Unknown serialization type: {type_}")
```

### 12.5.3 Msgpack Extension Type 机制

ormsgpack 默认不支持 Python 的许多类型。JsonPlusSerializer 通过 Extension Type 机制扩展：

```python
# Extension Type 编号
EXT_CONSTRUCTOR_SINGLE_ARG = 0   # cls(arg)
EXT_CONSTRUCTOR_POS_ARGS = 1     # cls(*args)
EXT_CONSTRUCTOR_KW_ARGS = 2      # cls(**kwargs)
EXT_METHOD_SINGLE_ARG = 3        # cls.method(arg)
EXT_PYDANTIC_V1 = 4              # Pydantic v1
EXT_PYDANTIC_V2 = 5              # Pydantic v2
EXT_NUMPY_ARRAY = 6              # NumPy 数组
```

序列化时，`_msgpack_default` 函数根据对象类型选择编码方式。每种支持的类型会被编码为 `(module, name, args/kwargs)` 元组，反序列化时通过 `importlib.import_module` 动态导入并重建对象：

```python
def _msgpack_default(obj: Any) -> str | ormsgpack.Ext:
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        # Pydantic v2
        return ormsgpack.Ext(EXT_PYDANTIC_V2, _msgpack_enc(
            (obj.__class__.__module__, obj.__class__.__name__,
             obj.model_dump(), "model_validate_json")))

    elif isinstance(obj, UUID):
        return ormsgpack.Ext(EXT_CONSTRUCTOR_SINGLE_ARG, _msgpack_enc(
            (obj.__class__.__module__, obj.__class__.__name__, obj.hex)))

    elif isinstance(obj, (set, frozenset, deque)):
        return ormsgpack.Ext(EXT_CONSTRUCTOR_SINGLE_ARG, _msgpack_enc(
            (obj.__class__.__module__, obj.__class__.__name__, tuple(obj))))

    elif isinstance(obj, datetime):
        return ormsgpack.Ext(EXT_METHOD_SINGLE_ARG, _msgpack_enc(
            (obj.__class__.__module__, obj.__class__.__name__,
             obj.isoformat(), "fromisoformat")))

    elif dataclasses.is_dataclass(obj):
        return ormsgpack.Ext(EXT_CONSTRUCTOR_KW_ARGS, _msgpack_enc(
            (obj.__class__.__module__, obj.__class__.__name__,
             {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})))

    elif isinstance(obj, BaseException):
        return repr(obj)  # 异常转为字符串表示

    # ... 还支持 pathlib.Path, re.Pattern, Decimal, IP 地址,
    #     ZoneInfo, Enum, Send, Item, NamedTuple, NumPy array 等
```

### 12.5.4 Msgpack 安全白名单

出于安全考虑，反序列化时只允许预定义的安全类型：

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py

SAFE_MSGPACK_TYPES: frozenset[tuple[str, ...]] = frozenset({
    # datetime 类型
    ("datetime", "datetime"), ("datetime", "date"),
    ("datetime", "time"), ("datetime", "timedelta"), ("datetime", "timezone"),
    # uuid
    ("uuid", "UUID"),
    # 集合类型
    ("builtins", "set"), ("builtins", "frozenset"), ("collections", "deque"),
    # IP 地址
    ("ipaddress", "IPv4Address"), ("ipaddress", "IPv4Network"),
    ("ipaddress", "IPv6Address"), ("ipaddress", "IPv6Network"),
    # ... 更多 ...
    # pathlib
    ("pathlib", "Path"), ("pathlib", "PosixPath"), ("pathlib", "WindowsPath"),
    # zoneinfo
    ("zoneinfo", "ZoneInfo"),
    # langchain-core messages
    ("langchain_core.messages.human", "HumanMessage"),
    ("langchain_core.messages.ai", "AIMessage"),
    ("langchain_core.messages.system", "SystemMessage"),
    ("langchain_core.messages.tool", "ToolMessage"),
    # ... 更多 langchain-core 类型 ...
    # langgraph 类型
    ("langgraph.types", "Send"), ("langgraph.types", "Interrupt"),
    ("langgraph.types", "Command"), ("langgraph.types", "Overwrite"),
    ("langgraph.store.base", "Item"), ("langgraph.store.base", "GetOp"),
})

SAFE_MSGPACK_METHODS: frozenset[tuple[str, str, str]] = frozenset({
    ("datetime", "datetime", "fromisoformat"),
})
```

严格模式通过环境变量控制：

```python
STRICT_MSGPACK_ENABLED = os.getenv("LANGGRAPH_STRICT_MSGPACK", "false").lower() in (
    "1", "true", "yes",
)
```

- **非严格模式**（默认）：不在白名单中的类型发出警告但允许反序列化
- **严格模式**：阻止反序列化，返回原始数据（dict/list 等）

### 12.5.5 自定义类型的白名单配置

```python
# 方式 1：创建 serializer 时指定
serde = JsonPlusSerializer(
    allowed_msgpack_modules=[("myapp.models", "UserProfile")]
)
saver = SqliteSaver(conn, serde=serde)

# 方式 2：使用 with_allowlist（推荐）
saver = SqliteSaver(conn).with_allowlist([
    ("myapp.models", "UserProfile"),
    ("myapp.models", "ChatSession"),
])

# 方式 3：允许所有类型（不推荐，有安全风险）
serde = JsonPlusSerializer(allowed_msgpack_modules=True)
```

### 12.5.6 EncryptedSerializer

`EncryptedSerializer` 在底层序列化器之上添加加密层：

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/serde/encrypted.py

class EncryptedSerializer(SerializerProtocol):
    def __init__(
        self, cipher: CipherProtocol, serde: SerializerProtocol = JsonPlusSerializer()
    ) -> None:
        self.cipher = cipher
        self.serde = serde

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        typ, data = self.serde.dumps_typed(obj)         # 先序列化
        ciphername, ciphertext = self.cipher.encrypt(data)  # 再加密
        return f"{typ}+{ciphername}", ciphertext            # 组合类型标识

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        enc_cipher, ciphertext = data
        if "+" not in enc_cipher:
            return self.serde.loads_typed(data)          # 兼容未加密数据
        typ, ciphername = enc_cipher.split("+", 1)
        decrypted_data = self.cipher.decrypt(ciphername, ciphertext)
        return self.serde.loads_typed((typ, decrypted_data))
```

类型标识变为 `"msgpack+aes"` 格式。`loads_typed` 通过 `+` 分隔符判断是否需要解密——未加密的旧数据可以被 EncryptedSerializer 直接读取。

#### AES 便捷方法

```python
# 从环境变量读取密钥
serde = EncryptedSerializer.from_pycryptodome_aes()  # 需 LANGGRAPH_AES_KEY 环境变量

# 手动指定密钥
serde = EncryptedSerializer.from_pycryptodome_aes(key=b"0123456789abcdef0123456789abcdef")

saver = PostgresSaver(conn, serde=serde)
```

`from_pycryptodome_aes` 内部使用 AES-EAX 模式（认证加密），密文格式为 `nonce(16) + tag(16) + ciphertext`。

### 12.5.7 CipherProtocol

```python
# 源码路径: libs/checkpoint/langgraph/checkpoint/serde/base.py

class CipherProtocol(Protocol):
    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        """返回 (密码名称, 密文)"""
        ...
    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        """返回明文"""
        ...
```

你可以实现自己的 `CipherProtocol` 来使用其他加密库或算法。

---

## 12.6 Serde 完整数据流

```
序列化 (存储 checkpoint):
  Python 对象
    -> JsonPlusSerializer.dumps_typed()
      -> ormsgpack.packb() + _msgpack_default()
        -> 自定义类型编码为 Extension Type (module, name, args)
      -> 返回 ("msgpack", bytes)
    -> [可选] EncryptedSerializer
      -> cipher.encrypt(bytes)
      -> 返回 ("msgpack+aes", encrypted_bytes)
    -> 存入数据库 (type 列 + blob 列)

反序列化 (加载 checkpoint):
  数据库行 (type, blob)
    -> [可选] EncryptedSerializer.loads_typed()
      -> 检测 "+" 分隔符
      -> cipher.decrypt(bytes)
      -> 得到 ("msgpack", decrypted_bytes)
    -> JsonPlusSerializer.loads_typed()
      -> ormsgpack.unpackb() + ext_hook()
        -> 安全白名单检查
        -> importlib.import_module + 构造函数重建对象
    -> Python 对象
```

---

## 本章要点

1. **InMemorySaver** 将所有数据存储在 Python 字典中，适合开发和测试。零依赖，无持久化

2. **SqliteSaver** 使用两张表（`checkpoints` + `writes`），checkpoint 以 BLOB 存储，metadata 以 JSON 存储。使用 WAL 模式和 `threading.Lock` 保证并发安全。版本号为零填充字符串加随机后缀

3. **PostgresSaver** 使用四张表（`checkpoint_migrations` + `checkpoints` + `checkpoint_blobs` + `checkpoint_writes`），将 channel_values 中的复杂值拆分到 blobs 表实现增量存储。支持 pipeline 模式、连接池和有序迁移系统

4. **JsonPlusSerializer** 是默认序列化器，基于 `ormsgpack`。通过 7 种 Extension Type 支持 datetime、UUID、set、Pydantic v1/v2 模型、dataclass、NumPy array、pathlib.Path、Enum 等多种 Python 类型

5. **安全白名单**：`SAFE_MSGPACK_TYPES` 定义了约 40 种允许反序列化的类型。严格模式通过 `LANGGRAPH_STRICT_MSGPACK` 环境变量启用。自定义类型需要通过 `allowed_msgpack_modules` 或 `with_allowlist` 显式注册

6. **EncryptedSerializer** 是装饰器模式的序列化器，在底层序列化器之上添加加密层。内置 AES-EAX 认证加密支持，向后兼容未加密的旧数据

7. **put_writes 的幂等性设计**：特殊写入（ERROR、INTERRUPT 等）使用 REPLACE 语义允许覆盖，普通写入使用 IGNORE 语义保证幂等性
