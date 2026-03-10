# 第 12 章 三种 Checkpointer 实现 + Serde

上一章介绍了 `BaseCheckpointSaver` 接口。本章深入三个官方实现——InMemorySaver、SqliteSaver、PostgresSaver——以及它们共同依赖的序列化层 Serde。

## InMemorySaver：开发调试利器

`InMemorySaver` 是最简单的 checkpointer，所有数据存储在 Python 字典中，进程退出即丢失。它是开发和测试的首选。

### 存储结构

```python
# libs/checkpoint/langgraph/checkpoint/memory/__init__.py

class InMemorySaver(
    BaseCheckpointSaver[str], AbstractContextManager, AbstractAsyncContextManager
):
    # thread ID -> checkpoint NS -> checkpoint ID -> checkpoint mapping
    storage: defaultdict[
        str,
        dict[str, dict[str, tuple[tuple[str, bytes], tuple[str, bytes], str | None]]],
    ]
    # (thread ID, checkpoint NS, checkpoint ID) -> (task ID, write idx)
    writes: defaultdict[
        tuple[str, str, str],
        dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]],
    ]
    blobs: dict[
        tuple[str, str, str, str | int | float],
        tuple[str, bytes],
    ]
```

三层嵌套字典对应 checkpoint 的寻址模型。关键设计：channel values 不内联在 checkpoint 中，而是按 `(thread_id, checkpoint_ns, channel, version)` 单独存放在 `blobs` 里。读取时通过版本号从 `blobs` 加载：

```python
# libs/checkpoint/langgraph/checkpoint/memory/__init__.py

def _load_blobs(
    self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
) -> dict[str, Any]:
    channel_values: dict[str, Any] = {}
    for k, v in versions.items():
        kk = (thread_id, checkpoint_ns, k, v)
        if kk in self.blobs:
            vv = self.blobs[kk]
            if vv[0] != "empty":
                channel_values[k] = self.serde.loads_typed(vv)
    return channel_values
```

多个 checkpoint 如果某个 channel 没有变化，它们共享同一份 blob 数据，节省存储空间。

### put 方法

```python
# libs/checkpoint/langgraph/checkpoint/memory/__init__.py

def put(self, config, checkpoint, metadata, new_versions) -> RunnableConfig:
    c = checkpoint.copy()
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
    values: dict[str, Any] = c.pop("channel_values")
    for k, v in new_versions.items():
        self.blobs[(thread_id, checkpoint_ns, k, v)] = (
            self.serde.dumps_typed(values[k]) if k in values else ("empty", b"")
        )
    self.storage[thread_id][checkpoint_ns].update({
        checkpoint["id"]: (
            self.serde.dumps_typed(c),
            self.serde.dumps_typed(get_checkpoint_metadata(config, metadata)),
            config["configurable"].get("checkpoint_id"),  # parent
        )
    })
```

流程：分离 channel_values，仅将 `new_versions` 中更新的 channel 写入 blobs，将精简后的 checkpoint 结构序列化存入 storage。

### put_writes 的幂等性

```python
# libs/checkpoint/langgraph/checkpoint/memory/__init__.py

for idx, (c, v) in enumerate(writes):
    inner_key = (task_id, WRITES_IDX_MAP.get(c, idx))
    if inner_key[1] >= 0 and outer_writes_ and inner_key in outer_writes_:
        continue
    self.writes[outer_key][inner_key] = (task_id, c, self.serde.dumps_typed(v), task_path)
```

正数索引的普通写入如果已存在就跳过（幂等性）。负数索引的特殊写入（error、interrupt 等）因 `inner_key[1] >= 0` 为 false 而总是写入，允许错误信息更新覆盖。

### 版本号策略

```python
# libs/checkpoint/langgraph/checkpoint/memory/__init__.py

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

格式 `"序号.随机数"` 保证单调递增，随机数部分降低冲突概率。源码尾部保留了 `MemorySaver = InMemorySaver` 的向后兼容别名。

## SqliteSaver：单机持久化

`SqliteSaver` 使用 SQLite 作为存储后端，适合单机场景。

### 表结构

```sql
-- libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/__init__.py
PRAGMA journal_mode=WAL;
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

`PRAGMA journal_mode=WAL` 启用 Write-Ahead Logging，允许并发读写。`threading.Lock` 保证线程安全：

```python
# libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/__init__.py

@contextmanager
def cursor(self, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
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

异步版本 `AsyncSqliteSaver` 基于 `aiosqlite` 库，位于 `langgraph.checkpoint.sqlite.aio` 模块中。

## PostgresSaver：生产级方案

PostgresSaver 基于 `psycopg` (v3) 和 `psycopg_pool` 构建，支持连接池和 Pipeline 优化。

```python
# libs/checkpoint-postgres/langgraph/checkpoint/postgres/__init__.py

class PostgresSaver(BasePostgresSaver):
    def __init__(self, conn: _internal.Conn, pipe: Pipeline | None = None, serde=None):
        super().__init__(serde=serde)
        if isinstance(conn, ConnectionPool) and pipe is not None:
            raise ValueError("Pipeline should be used only with a single Connection, not ConnectionPool.")
        self.conn = conn
        self.pipe = pipe
        self.lock = threading.Lock()
```

PostgreSQL 表结构使用迁移系统管理，checkpoint 和 metadata 使用 JSONB 类型（允许数据库层查询），channel 值存放在独立的 `checkpoint_blobs` 表中（BYTEA 类型）。

### ShallowPostgresSaver：浅层 checkpoint

生产环境中保留全部历史 checkpoint 会导致存储膨胀。`ShallowPostgresSaver` 只保留最新 checkpoint：

```sql
-- libs/checkpoint-postgres/langgraph/checkpoint/postgres/shallow.py
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns)  -- 无 checkpoint_id，每次 UPSERT 覆盖
);
CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel)  -- 无 version，只保留最新值
);
```

代价是失去 time-travel 能力，适用于只需要最新状态的场景。

## Serde：序列化体系

### SerializerProtocol

```python
# libs/checkpoint/langgraph/checkpoint/serde/base.py

@runtime_checkable
class SerializerProtocol(Protocol):
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...
    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...
```

返回 `(type_tag, bytes)` 二元组，type_tag 标识格式（`"msgpack"`、`"json"`、`"bytes"`、`"null"`）。

### JsonPlusSerializer

默认序列化器，优先使用 msgpack：

```python
# libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py

class JsonPlusSerializer(SerializerProtocol):
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

通过 7 种 `ormsgpack.Ext` 扩展类型代码处理 Python 复杂类型：

```python
EXT_CONSTRUCTOR_SINGLE_ARG = 0   # set, frozenset, UUID, Decimal, Enum...
EXT_CONSTRUCTOR_POS_ARGS = 1     # Path, re.Pattern, timedelta, Send...
EXT_CONSTRUCTOR_KW_ARGS = 2      # namedtuple, dataclass, time...
EXT_METHOD_SINGLE_ARG = 3        # datetime (via fromisoformat)
EXT_PYDANTIC_V1 = 4              # Pydantic v1 模型
EXT_PYDANTIC_V2 = 5              # Pydantic v2 模型
EXT_NUMPY_ARRAY = 6              # NumPy 数组（保留 dtype/shape/layout）
```

每种扩展都序列化为 `(module, class_name, data[, method])` 元组，反序列化时动态 import 并构造。

### 安全性：allowlist 机制

反序列化涉及动态 import，存在安全风险。`SAFE_MSGPACK_TYPES` 定义了无条件安全的类型白名单：

```python
# libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py

SAFE_MSGPACK_TYPES: frozenset[tuple[str, ...]] = frozenset({
    ("datetime", "datetime"),
    ("uuid", "UUID"),
    ("decimal", "Decimal"),
    ("builtins", "set"),
    ("langchain_core.messages.ai", "AIMessage"),
    ("langchain_core.messages.human", "HumanMessage"),
    ("langgraph.types", "Send"),
    ("langgraph.store.base", "Item"),
    # ... 更多安全类型
})
```

不在白名单中的类型需要通过 `allowed_msgpack_modules` 参数授权。环境变量 `LANGGRAPH_STRICT_MSGPACK=true` 启用严格模式，拒绝未注册类型。

### EncryptedSerializer

在序列化之上叠加加密层：

```python
# libs/checkpoint/langgraph/checkpoint/serde/encrypted.py

class EncryptedSerializer(SerializerProtocol):
    def __init__(self, cipher: CipherProtocol, serde: SerializerProtocol = JsonPlusSerializer()):
        self.cipher = cipher
        self.serde = serde

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        typ, data = self.serde.dumps_typed(obj)
        ciphername, ciphertext = self.cipher.encrypt(data)
        return f"{typ}+{ciphername}", ciphertext

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        enc_cipher, ciphertext = data
        if "+" not in enc_cipher:
            return self.serde.loads_typed(data)  # 未加密数据直接透传
        typ, ciphername = enc_cipher.split("+", 1)
        decrypted_data = self.cipher.decrypt(ciphername, ciphertext)
        return self.serde.loads_typed((typ, decrypted_data))
```

type tag 通过 `+` 组合（如 `"msgpack+aes"`）。对不含 `+` 的标签直接透传，支持加密与未加密数据混合读取。`from_pycryptodome_aes` 工厂方法提供开箱即用的 AES-EAX 加密，密钥可从 `LANGGRAPH_AES_KEY` 环境变量读取。

## 三种实现的对比

| 特性 | InMemorySaver | SqliteSaver | PostgresSaver |
|------|--------------|-------------|---------------|
| 持久化 | 否 | 是（文件） | 是（数据库） |
| 并发安全 | 单进程 | 单进程（锁） | 多进程（连接池） |
| 异步支持 | 同步回调 | aiosqlite | 原生 psycopg async |
| 浅层模式 | 否 | 否 | ShallowPostgresSaver |
| 适用场景 | 开发/测试 | 单机/Demo | 生产环境 |

## 本章要点

1. **InMemorySaver** 使用三层嵌套字典，channel values 按版本独立存放在 `blobs` 中，多个 checkpoint 可共享未变化的 channel 数据。
2. **SqliteSaver** 使用两张表和 `threading.Lock`，`WAL` 模式优化并发读写，异步版本基于 aiosqlite。
3. **PostgresSaver** 支持连接池和 Pipeline。**ShallowPostgresSaver** 只保留最新 checkpoint，用 time-travel 能力换取存储效率。
4. **JsonPlusSerializer** 优先使用 msgpack，通过 7 种扩展类型代码支持 Pydantic 模型、datetime、NumPy 数组等 Python 复杂类型。
5. **allowlist 机制**控制反序列化安全性，`SAFE_MSGPACK_TYPES` 白名单覆盖 LangChain 消息类型和 LangGraph 核心类型。
6. **EncryptedSerializer** 通过组合模式叠加加密，type tag 使用 `+` 分隔，支持加密与未加密数据的混合读取。
