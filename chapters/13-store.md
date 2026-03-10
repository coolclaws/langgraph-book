# 第 13 章 Store：跨线程持久化键值存储

Checkpoint 解决了单个线程内的状态持久化问题，但有些数据天然需要跨线程共享——用户偏好、长期记忆、全局配置、RAG 文档索引等。LangGraph 的 Store 子系统正是为此而生。本章将深入 `langgraph.store.base` 和 `langgraph.store.memory` 两个模块，剖析 Store 的接口设计、数据模型和实现细节。

---

## 13.1 Store 与 Checkpoint 的本质区别

在深入 Store 的源码之前，先厘清它与 Checkpoint 的关系：

| 维度 | Checkpoint | Store |
|------|-----------|-------|
| 隔离范围 | `thread_id` + `checkpoint_ns` | 无固定隔离，由 namespace 自由组织 |
| 核心用途 | 保存/恢复图的执行状态 | 持久化键值数据，跨线程共享 |
| 数据生命周期 | 与 thread 绑定 | 独立于 thread |
| 访问方式 | 由 Pregel 引擎自动管理 | 由节点代码显式调用 |
| 搜索能力 | 按 thread_id/checkpoint_id 查询 | 支持 filter、语义搜索 |
| 典型数据 | channel_values, versions_seen | 用户偏好、长期记忆、文档 |

关键区别在于**隔离范围**：Checkpoint 被 `thread_id` 严格隔离，不同线程看不到彼此的 checkpoint。Store 则没有这种限制——通过精心设计 namespace，一个节点可以访问任何线程写入的数据。

例如，用户 Alice 在 thread-1 中存储了偏好 `{"theme": "dark"}`，在 thread-2 中可以通过相同的 namespace `("users", "alice")` 读取这个偏好。这在 Checkpoint 模型下是不可能的。

---

## 13.2 BaseStore 抽象接口

`BaseStore` 是所有 Store 实现的基类，定义在：

```python
# 源码路径: libs/checkpoint/langgraph/store/base/__init__.py

class BaseStore(ABC):
    """Abstract base class for persistent key-value stores.

    Stores enable persistence and memory that can be shared across threads,
    scoped to user IDs, assistant IDs, or other arbitrary namespaces.
    Some implementations may support semantic search capabilities through
    an optional index configuration.
    """

    supports_ttl: bool = False
    ttl_config: TTLConfig | None = None

    __slots__ = ("__weakref__",)

    @abstractmethod
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations synchronously in a single batch."""

    @abstractmethod
    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously in a single batch."""
```

BaseStore 的核心设计理念是 **batch-first**——所有操作最终都通过 `batch` / `abatch` 方法执行，而 `get`、`put`、`search` 等便捷方法只是构建对应的 Op 对象然后调用 `batch`。这种设计使得实现类可以对多个操作进行批量优化（如批量 embedding 计算）。

### 13.2.1 核心方法概览

BaseStore 提供了五组核心操作：

#### get — 精确获取

```python
def get(
    self,
    namespace: tuple[str, ...],
    key: str,
    *,
    refresh_ttl: bool | None = None,
) -> Item | None:
    """Retrieve a single item."""
    return self.batch(
        [GetOp(namespace, str(key), _ensure_refresh(self.ttl_config, refresh_ttl))]
    )[0]
```

通过 `(namespace, key)` 精确获取一个 Item。如果启用了 TTL，`refresh_ttl=True` 会刷新过期时间。

#### put — 存储/更新

```python
def put(
    self,
    namespace: tuple[str, ...],
    key: str,
    value: dict[str, Any],
    index: Literal[False] | list[str] | None = None,
    *,
    ttl: float | None | NotProvided = NOT_PROVIDED,
) -> None:
    """Store or update an item in the store."""
    _validate_namespace(namespace)
    if ttl not in (NOT_PROVIDED, None) and not self.supports_ttl:
        raise NotImplementedError(
            f"TTL is not supported by {self.__class__.__name__}."
        )
    self.batch([PutOp(namespace, str(key), value, index=index,
                       ttl=_ensure_ttl(self.ttl_config, ttl))])
```

`index` 参数控制语义搜索索引：
- `None`：使用 store 默认的索引配置
- `False`：禁用索引（该项不可被语义搜索找到，但仍可通过 `get` 访问）
- `list[str]`：指定要索引的字段路径（如 `["text", "metadata.title"]`）

#### search — 搜索

```python
def search(
    self,
    namespace_prefix: tuple[str, ...],
    /,
    *,
    query: str | None = None,
    filter: dict[str, Any] | None = None,
    limit: int = 10,
    offset: int = 0,
    refresh_ttl: bool | None = None,
) -> list[SearchItem]:
    """Search for items within a namespace prefix."""
    return self.batch([SearchOp(
        namespace_prefix, filter, limit, offset, query,
        _ensure_refresh(self.ttl_config, refresh_ttl)
    )])[0]
```

搜索支持两种模式：
- **过滤搜索**：通过 `filter` 参数按 value 中的字段精确匹配或比较
- **语义搜索**：通过 `query` 参数进行自然语言相似度搜索（需要配置 embedding）

两种模式可以组合使用。

#### delete — 删除

```python
def delete(self, namespace: tuple[str, ...], key: str) -> None:
    """Delete an item."""
    self.batch([PutOp(namespace, str(key), None, ttl=None)])
```

删除通过 `PutOp` 实现——将 `value` 设为 `None` 即表示删除。

#### list_namespaces — 列出命名空间

```python
def list_namespaces(
    self,
    *,
    prefix: NamespacePath | None = None,
    suffix: NamespacePath | None = None,
    max_depth: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[tuple[str, ...]]:
    """List and filter namespaces in the store."""
    match_conditions = []
    if prefix:
        match_conditions.append(MatchCondition(match_type="prefix", path=prefix))
    if suffix:
        match_conditions.append(MatchCondition(match_type="suffix", path=suffix))
    op = ListNamespacesOp(
        match_conditions=tuple(match_conditions),
        max_depth=max_depth, limit=limit, offset=offset,
    )
    return self.batch([op])[0]
```

支持前缀匹配和后缀匹配，通配符 `"*"` 可以匹配任意单个层级。

### 13.2.2 异步方法

每个同步方法都有对应的异步版本：

```python
async def aget(self, namespace, key, *, refresh_ttl=None) -> Item | None: ...
async def aput(self, namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None: ...
async def asearch(self, namespace_prefix, /, *, query, filter, limit, offset) -> list[SearchItem]: ...
async def adelete(self, namespace, key) -> None: ...
async def alist_namespaces(self, *, prefix, suffix, max_depth, limit, offset) -> list: ...
```

---

## 13.3 Op 类型系统

Store 的所有操作都被建模为 NamedTuple 类型的 Op 对象：

### 13.3.1 GetOp — 获取操作

```python
class GetOp(NamedTuple):
    namespace: tuple[str, ...]
    key: str
    refresh_ttl: bool = True
```

### 13.3.2 SearchOp — 搜索操作

```python
class SearchOp(NamedTuple):
    namespace_prefix: tuple[str, ...]
    filter: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0
    query: str | None = None
    refresh_ttl: bool = True
```

`filter` 支持精确匹配和比较运算符：

```python
# 精确匹配
{"status": "active"}

# 比较运算符
{"score": {"$gt": 4.99}}     # 大于
{"score": {"$gte": 3.0}}     # 大于等于
{"score": {"$lt": 10.0}}     # 小于
{"score": {"$lte": 5.0}}     # 小于等于
{"status": {"$ne": "deleted"}}  # 不等于
{"status": {"$eq": "active"}}   # 等于（等价于精确匹配）

# 多条件组合
{"score": {"$gte": 3.0}, "color": "red"}
```

### 13.3.3 PutOp — 写入/删除操作

```python
class PutOp(NamedTuple):
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any] | None  # None 表示删除
    index: Literal[False] | list[str] | None = None
    ttl: float | None = None
```

`index` 参数支持丰富的路径语法：

```python
# 简单字段
["text", "summary"]

# 嵌套字段
["metadata.title", "content.body"]

# 数组索引
["array[0]"]       # 第一个元素
["array[-1]"]      # 最后一个元素
["array[*]"]       # 所有元素（每个独立索引）

# 深层嵌套
["messages[*].content"]             # 每条消息的 content
["sections[*].paragraphs[*].text"]  # 所有段落的文本
```

### 13.3.4 ListNamespacesOp — 列出命名空间操作

```python
class ListNamespacesOp(NamedTuple):
    match_conditions: tuple[MatchCondition, ...] | None = None
    max_depth: int | None = None
    limit: int = 100
    offset: int = 0

class MatchCondition(NamedTuple):
    match_type: NamespaceMatchType  # "prefix" | "suffix"
    path: NamespacePath             # tuple[str | Literal["*"], ...]
```

### 13.3.5 Op 和 Result 的统一类型

```python
Op = GetOp | SearchOp | PutOp | ListNamespacesOp
Result = Item | list[Item] | list[SearchItem] | list[tuple[str, ...]] | None
```

`batch` 方法接受 `Iterable[Op]`，返回 `list[Result]`，结果顺序与输入操作一一对应。

---

## 13.4 Item 数据模型

`Item` 是 Store 中的基本数据单元：

```python
# 源码路径: libs/checkpoint/langgraph/store/base/__init__.py

class Item:
    """Represents a stored item with metadata."""

    __slots__ = ("value", "key", "namespace", "created_at", "updated_at")

    def __init__(
        self,
        *,
        value: dict[str, Any],
        key: str,
        namespace: tuple[str, ...],
        created_at: datetime,
        updated_at: datetime,
    ):
        self.value = value
        self.key = key
        self.namespace = tuple(namespace)
        self.created_at = (
            datetime.fromisoformat(cast(str, created_at))
            if isinstance(created_at, str) else created_at
        )
        self.updated_at = (
            datetime.fromisoformat(cast(str, updated_at))
            if isinstance(updated_at, str) else updated_at
        )
```

各字段含义：

| 字段 | 类型 | 说明 |
|------|------|------|
| `value` | `dict[str, Any]` | 存储的数据，键可被 filter 搜索 |
| `key` | `str` | namespace 内的唯一标识符 |
| `namespace` | `tuple[str, ...]` | 层次化的命名空间路径 |
| `created_at` | `datetime` | 创建时间 |
| `updated_at` | `datetime` | 最后更新时间 |

注意 `value` 必须是 `dict[str, Any]`，不支持存储原始类型。这是因为 filter 搜索需要按字段名匹配。

### 13.4.1 SearchItem — 搜索结果

```python
class SearchItem(Item):
    """Represents an item returned from a search operation with additional metadata."""

    __slots__ = ("score",)

    def __init__(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        created_at: datetime,
        updated_at: datetime,
        score: float | None = None,
    ) -> None:
        super().__init__(
            value=value, key=key, namespace=namespace,
            created_at=created_at, updated_at=updated_at,
        )
        self.score = score
```

`SearchItem` 继承 `Item`，额外包含 `score` 字段——语义搜索时的相似度分数。如果搜索未使用 `query` 参数，`score` 为 `None`。

---

## 13.5 Namespace 层次结构

Namespace 是 Store 中最重要的组织概念。它是一个字符串元组，类似文件系统的目录路径：

```python
# 用户偏好
("users", "alice", "preferences")

# 对话记忆
("memories", "user-123", "conversations")

# 全局配置
("config", "global")

# RAG 文档
("documents", "knowledge-base", "v2")
```

### 13.5.1 Namespace 设计模式

**按用户隔离**：

```python
# 每个用户的数据在独立的 namespace 下
store.put(("users", user_id, "profile"), "info", {"name": "Alice", "role": "admin"})
store.put(("users", user_id, "preferences"), "theme", {"dark_mode": True})

# 搜索某个用户的所有数据
results = store.search(("users", user_id))
```

**按功能组织**：

```python
# 长期记忆
store.put(("memory", user_id), "fact_1", {"content": "User prefers Python"})

# 文档索引
store.put(("docs", "technical"), "doc_1", {"text": "LangGraph architecture..."})
```

**跨线程共享**：

```python
# 在 thread-1 中存储
def node_in_thread_1(state, *, store):
    store.put(("shared", "results"), "analysis", {"score": 0.95})

# 在 thread-2 中读取
def node_in_thread_2(state, *, store):
    item = store.get(("shared", "results"), "analysis")
    # item.value == {"score": 0.95}
```

### 13.5.2 list_namespaces 和通配符

`list_namespaces` 支持 `prefix` 和 `suffix` 过滤，以及通配符 `"*"`：

```python
# 列出所有用户的 namespace
store.list_namespaces(prefix=("users",))
# [("users", "alice"), ("users", "bob"), ("users", "charlie")]

# 列出所有以 "v2" 结尾的 namespace
store.list_namespaces(suffix=("v2",))
# [("documents", "knowledge-base", "v2"), ("config", "v2")]

# 通配符：匹配任意单个层级
store.list_namespaces(prefix=("users", "*", "preferences"))
# [("users", "alice", "preferences"), ("users", "bob", "preferences")]

# max_depth 截断
store.list_namespaces(prefix=("users",), max_depth=2)
# [("users", "alice"), ("users", "bob")]  # 不返回更深的层级
```

---

## 13.6 InMemoryStore 实现

`InMemoryStore` 是最简单的 Store 实现，将所有数据存储在 Python 字典中：

```python
# 源码路径: libs/checkpoint/langgraph/store/memory/__init__.py

class InMemoryStore(BaseStore):
    """In-memory dictionary-backed store with optional vector search."""

    __slots__ = ("_data", "_vectors", "index_config", "embeddings")

    def __init__(self, *, index: IndexConfig | None = None) -> None:
        self._data: dict[tuple[str, ...], dict[str, Item]] = defaultdict(dict)
        self._vectors: dict[tuple[str, ...], dict[str, dict[str, list[float]]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        self.index_config = index
        if self.index_config:
            self.index_config = self.index_config.copy()
            self.embeddings: Embeddings | None = ensure_embeddings(
                self.index_config.get("embed"),
            )
            self.index_config["__tokenized_fields"] = [
                (p, tokenize_path(p)) if p != "$" else (p, p)
                for p in (self.index_config.get("fields") or ["$"])
            ]
        else:
            self.index_config = None
            self.embeddings = None
```

### 13.6.1 内部数据结构

```python
# _data: namespace -> key -> Item
self._data = defaultdict(dict)
# 示例:
# {
#     ("users", "alice"): {
#         "profile": Item(value={"name": "Alice"}, key="profile", ...),
#         "prefs": Item(value={"theme": "dark"}, key="prefs", ...),
#     },
#     ("docs",): {
#         "doc1": Item(value={"text": "..."}, key="doc1", ...),
#     },
# }

# _vectors: namespace -> key -> field_path -> embedding_vector
self._vectors = defaultdict(lambda: defaultdict(dict))
# 示例:
# {
#     ("docs",): {
#         "doc1": {
#             "text": [0.1, 0.2, 0.3, ...],  # 1536 维向量
#         },
#     },
# }
```

### 13.6.2 batch 方法

```python
def batch(self, ops: Iterable[Op]) -> list[Result]:
    results, put_ops, search_ops = self._prepare_ops(ops)
    if search_ops:
        queryinmem_store = self._embed_search_queries(search_ops)
        self._batch_search(search_ops, queryinmem_store, results)
    to_embed = self._extract_texts(put_ops)
    if to_embed and self.index_config and self.embeddings:
        embeddings = self.embeddings.embed_documents(list(to_embed))
        self._insertinmem_store(to_embed, embeddings)
    self._apply_put_ops(put_ops)
    return results
```

`batch` 方法的执行流程：

1. **`_prepare_ops`**：遍历所有 Op，对 GetOp 直接查字典取结果，对 SearchOp 和 PutOp 分别收集
2. **搜索处理**：如果有 SearchOp，先对 query 进行 embedding，然后执行批量搜索
3. **索引处理**：对需要索引的 PutOp，提取文本并计算 embedding
4. **写入处理**：应用所有 PutOp（新增、更新或删除）

### 13.6.3 _prepare_ops — 操作预处理

```python
def _prepare_ops(self, ops):
    results = []
    put_ops = {}
    search_ops = {}
    for i, op in enumerate(ops):
        if isinstance(op, GetOp):
            item = self._data[op.namespace].get(op.key)
            results.append(item)
        elif isinstance(op, SearchOp):
            search_ops[i] = (op, self._filter_items(op))
            results.append(None)  # 占位，后续填充
        elif isinstance(op, ListNamespacesOp):
            results.append(self._handle_list_namespaces(op))
        elif isinstance(op, PutOp):
            put_ops[(op.namespace, op.key)] = op
            results.append(None)
    return results, put_ops, search_ops
```

注意 `put_ops` 使用 `(namespace, key)` 为键——如果同一批次中有多个对同一项的写入，后面的会覆盖前面的。

### 13.6.4 _filter_items — 过滤匹配

```python
def _filter_items(self, op: SearchOp) -> list[tuple[Item, list[list[float]]]]:
    namespace_prefix = op.namespace_prefix
    filtered = []
    for namespace in self._data:
        if not (namespace[:len(namespace_prefix)] == namespace_prefix
                if len(namespace) >= len(namespace_prefix) else False):
            continue
        for key, item in self._data[namespace].items():
            if filter_func(item):  # 基于 op.filter 过滤
                if op.query and (embeddings := self._vectors[namespace].get(key)):
                    filtered.append((item, list(embeddings.values())))
                else:
                    filtered.append((item, []))
    return filtered
```

过滤逻辑先按 namespace 前缀筛选，再按 `filter` 条件过滤 value 中的字段。

### 13.6.5 语义搜索：余弦相似度

```python
def _cosine_similarity(X: list[float], Y: list[list[float]]) -> list[float]:
    if not Y:
        return []
    if _check_numpy():
        import numpy as np
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        X_norm = np.linalg.norm(X_arr)
        Y_norm = np.linalg.norm(Y_arr, axis=1)
        mask = Y_norm != 0
        similarities = np.zeros_like(Y_norm)
        similarities[mask] = np.dot(Y_arr[mask], X_arr) / (Y_norm[mask] * X_norm)
        return similarities.tolist()

    # 纯 Python 回退实现
    similarities = []
    for y in Y:
        dot_product = sum(a * b for a, b in zip(X, y, strict=False))
        norm1 = sum(a * a for a in X) ** 0.5
        norm2 = sum(a * a for a in y) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        similarities.append(similarity)
    return similarities
```

InMemoryStore 提供了两种余弦相似度实现：
- **NumPy 版本**：高性能，推荐安装 numpy
- **纯 Python 版本**：无依赖的回退方案，性能较低

`_check_numpy()` 使用 `@functools.lru_cache` 缓存检查结果，只检查一次。

### 13.6.6 搜索结果的 Max Pooling

当一个 Item 有多个字段被索引（生成多个向量）时，`_batch_search` 使用 max pooling 策略——取该 Item 所有向量与 query 的最高相似度作为该 Item 的分数：

```python
# 在 _batch_search 中
sorted_results = sorted(
    zip(scores, flat_items, strict=False),
    key=lambda x: x[0],
    reverse=True,
)
# max pooling: 每个 (namespace, key) 只保留最高分
seen: set[tuple[tuple[str, ...], str]] = set()
kept = []
for score, item in sorted_results:
    key = (item.namespace, item.key)
    if key in seen:
        continue
    seen.add(key)
    kept.append((score, item))
```

### 13.6.7 _apply_put_ops — 写入应用

```python
def _apply_put_ops(self, put_ops):
    for (namespace, key), op in put_ops.items():
        if op.value is None:
            self._data[namespace].pop(key, None)
            self._vectors[namespace].pop(key, None)
        else:
            self._data[namespace][key] = Item(
                value=op.value,
                key=key,
                namespace=namespace,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
```

删除操作（`value is None`）同时清理数据和向量索引。

---

## 13.7 IndexConfig：语义搜索配置

InMemoryStore 的语义搜索通过 `IndexConfig` 配置：

```python
# 源码路径: libs/checkpoint/langgraph/store/base/__init__.py

class IndexConfig(TypedDict, total=False):
    dims: int
    """Embedding 向量的维度数"""
    embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str
    """Embedding 函数或 LangChain Embeddings 实例"""
    fields: list[str] | None
    """要索引的字段路径列表，默认 ["$"]（整个文档）"""
```

配置示例：

```python
# 使用 LangChain embedding
from langchain.embeddings import init_embeddings

store = InMemoryStore(index={
    "dims": 1536,
    "embed": init_embeddings("openai:text-embedding-3-small"),
    "fields": ["text", "summary"],
})

# 使用自定义函数
from openai import OpenAI
client = OpenAI()

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small", input=texts
    )
    return [e.embedding for e in response.data]

store = InMemoryStore(index={
    "dims": 1536,
    "embed": embed_texts,
})

# 使用异步函数
from openai import AsyncOpenAI
async_client = AsyncOpenAI()

async def aembed_texts(texts: list[str]) -> list[list[float]]:
    response = await async_client.embeddings.create(
        model="text-embedding-3-small", input=texts
    )
    return [e.embedding for e in response.data]

store = InMemoryStore(index={
    "dims": 1536,
    "embed": aembed_texts,
})
```

`fields` 参数默认为 `["$"]`，表示将整个 value 字典序列化为文本后索引。指定具体字段可以让索引更精确：

```python
store = InMemoryStore(index={
    "dims": 1536,
    "embed": embed_func,
    "fields": ["content", "metadata.title"],  # 只索引这两个字段
})
```

---

## 13.8 TTL 支持

BaseStore 定义了 TTL（Time-To-Live）配置：

```python
class TTLConfig(TypedDict, total=False):
    refresh_on_read: bool
    """读操作是否刷新 TTL，默认 True"""
    default_ttl: float | None
    """新 item 的默认 TTL（分钟），None 表示不过期"""
    sweep_interval_minutes: int | None
    """清理过期 item 的间隔（分钟）"""
```

TTL 支持需要 store 实现明确声明 `supports_ttl = True`。InMemoryStore 不支持 TTL，但 PostgresStore 和 SqliteStore 的 store 实现支持。

---

## 13.9 Filter 比较运算符

`_compare_values` 和 `_apply_operator` 函数实现了类似 MongoDB/PostgreSQL JSONB 的过滤语义：

```python
# 源码路径: libs/checkpoint/langgraph/store/memory/__init__.py

def _compare_values(item_value: Any, filter_value: Any) -> bool:
    if isinstance(filter_value, dict):
        if any(k.startswith("$") for k in filter_value):
            return all(
                _apply_operator(item_value, op_key, op_value)
                for op_key, op_value in filter_value.items()
            )
        if not isinstance(item_value, dict):
            return False
        return all(
            _compare_values(item_value.get(k), v) for k, v in filter_value.items()
        )
    elif isinstance(filter_value, (list, tuple)):
        return (isinstance(item_value, (list, tuple))
                and len(item_value) == len(filter_value)
                and all(_compare_values(iv, fv) for iv, fv in zip(item_value, filter_value)))
    else:
        return item_value == filter_value


def _apply_operator(value: Any, operator: str, op_value: Any) -> bool:
    if operator == "$eq":
        return value == op_value
    elif operator == "$gt":
        return float(value) > float(op_value)
    elif operator == "$gte":
        return float(value) >= float(op_value)
    elif operator == "$lt":
        return float(value) < float(op_value)
    elif operator == "$lte":
        return float(value) <= float(op_value)
    elif operator == "$ne":
        return value != op_value
    else:
        raise ValueError(f"Unsupported operator: {operator}")
```

支持嵌套字典的深层比较和六种比较运算符。

---

## 13.10 Namespace 匹配机制

`_does_match` 函数实现了通配符匹配：

```python
def _does_match(match_condition: MatchCondition, key: tuple[str, ...]) -> bool:
    match_type = match_condition.match_type
    path = match_condition.path

    if len(key) < len(path):
        return False

    if match_type == "prefix":
        for k_elem, p_elem in zip(key, path, strict=False):
            if p_elem == "*":
                continue  # 通配符匹配任意元素
            if k_elem != p_elem:
                return False
        return True
    elif match_type == "suffix":
        for k_elem, p_elem in zip(reversed(key), reversed(path), strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
```

前缀匹配从左到右，后缀匹配从右到左。`"*"` 通配符匹配任意单个层级。

---

## 13.11 数据库支持的 Store 实现

除了 InMemoryStore，LangGraph 还提供了数据库支持的 Store 实现：

### 13.11.1 PostgresStore

位于 `libs/checkpoint-postgres/langgraph/store/postgres/`，将数据存储在 PostgreSQL 中，支持：
- 持久化存储
- 基于 pgvector 的向量搜索
- JSONB 过滤
- TTL 过期
- 连接池

### 13.11.2 SqliteStore

位于 `libs/checkpoint-sqlite/langgraph/store/sqlite/`，将数据存储在 SQLite 中。

这些实现的接口与 InMemoryStore 完全一致（都继承 BaseStore），只是底层存储引擎不同。

---

## 13.12 在图中使用 Store

Store 在图的节点中通过依赖注入使用：

```python
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore

# 创建 store
store = InMemoryStore(index={
    "dims": 1536,
    "embed": embed_func,
})

# 在节点中使用 store（通过参数注入）
def my_node(state, *, store: BaseStore):
    # 读取用户偏好
    prefs = store.get(("users", state["user_id"]), "preferences")

    # 搜索相关记忆
    memories = store.search(
        ("memories", state["user_id"]),
        query=state["current_question"],
        limit=5,
    )

    # 存储新的记忆
    store.put(
        ("memories", state["user_id"]),
        f"memory_{uuid4().hex[:8]}",
        {"content": state["assistant_response"], "topic": "general"},
    )

    return {"context": [m.value for m in memories]}

# 编译图时传入 store
builder = StateGraph(State)
builder.add_node("my_node", my_node)
graph = builder.compile(store=store, checkpointer=InMemorySaver())
```

---

## 13.13 设计思考

### 13.13.1 为什么 value 必须是 dict

Store 要求 `value` 是 `dict[str, Any]`，不支持存储字符串、数字等原始类型。这是因为：

1. **filter 搜索**需要按字段名访问值
2. **语义索引**需要从 value 中提取特定字段的文本
3. **dict 是最通用的结构化数据容器**，适合存储各种类型的数据

如果需要存储简单值，可以包装为 dict：`{"value": "my string"}`。

### 13.13.2 为什么 delete 通过 PutOp 实现

```python
def delete(self, namespace, key):
    self.batch([PutOp(namespace, str(key), None, ttl=None)])
```

将删除建模为"写入 None"有几个好处：
- **batch 统一性**：所有操作都通过 `batch` 执行，简化实现
- **原子性**：在同一个 batch 中混合写入和删除时，操作的执行顺序是确定的
- **简化 API**：不需要单独的 DeleteOp 类型

### 13.13.3 Batch-first 设计的优势

所有高层方法（`get`、`put`、`search` 等）都委托给 `batch`：

```python
def get(self, namespace, key, *, refresh_ttl=None):
    return self.batch([GetOp(namespace, str(key), ...)])[0]
```

这意味着实现类只需要实现 `batch` 和 `abatch` 两个方法。优势包括：
- **批量优化**：可以合并 embedding 计算，减少 API 调用
- **事务语义**：数据库实现可以在一个事务中执行多个操作
- **简化实现**：只需关注批量执行路径

---

## 本章要点

1. **Store 与 Checkpoint 的核心区别**：Checkpoint 被 `thread_id` 严格隔离，Store 通过 namespace 自由组织，支持跨线程共享数据

2. **BaseStore** 采用 batch-first 设计——所有操作通过 `batch`/`abatch` 执行，`get`/`put`/`search`/`delete`/`list_namespaces` 是便捷方法

3. **四种 Op 类型**：`GetOp`（精确获取）、`SearchOp`（过滤 + 语义搜索）、`PutOp`（写入/删除，value=None 表示删除）、`ListNamespacesOp`（列出命名空间）

4. **Namespace 层次结构**：字符串元组构成的树形路径，支持前缀/后缀匹配和 `"*"` 通配符。通过 `list_namespaces` 探索数据组织结构

5. **InMemoryStore** 使用 `defaultdict(dict)` 存储数据，`defaultdict(lambda: defaultdict(dict))` 存储向量索引。支持可选的语义搜索（通过 `IndexConfig` 配置 embedding）

6. **语义搜索**：通过 `IndexConfig` 配置 embedding 函数和索引字段。搜索时使用余弦相似度，对多向量 Item 使用 max pooling 策略。优先使用 NumPy 加速，有纯 Python 回退

7. **Filter 运算符**：支持 `$eq`、`$ne`、`$gt`、`$gte`、`$lt`、`$lte` 六种比较运算符，以及嵌套字典的深层匹配。语法类似 MongoDB 查询
