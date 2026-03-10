# 第 13 章 Store：跨线程持久化键值存储

Checkpoint 解决了单个线程内的状态持久化，但有些数据天然需要跨线程共享——用户偏好、长期记忆、全局配置等。LangGraph 的 Store 子系统正是为此而生。

## Store 与 Checkpoint 的本质区别

| 维度 | Checkpoint | Store |
|------|-----------|-------|
| 隔离单位 | 绑定 `thread_id` + `checkpoint_ns` | 以 `namespace` 组织，跨 `thread_id` |
| 数据模型 | 图状态快照（channel values） | 通用键值对 |
| 时间维度 | 保留历史（可 time-travel） | 只保留最新值 |
| 读写时机 | Pregel 循环自动管理 | 节点内手动调用 |
| 典型用途 | 对话状态、中间结果 | 用户画像、长期记忆、跨会话知识 |

简言之，Checkpoint 是图的"工作记忆"，Store 是图的"长期记忆"。

## Item：存储的基本单元

```python
# libs/checkpoint/langgraph/store/base/__init__.py

class Item:
    __slots__ = ("value", "key", "namespace", "created_at", "updated_at")

    def __init__(
        self, *, value: dict[str, Any], key: str,
        namespace: tuple[str, ...], created_at: datetime, updated_at: datetime,
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

- **`value`**：数据本体，JSON 兼容字典，key 可用于过滤查询。
- **`key`**：命名空间内的唯一标识符。
- **`namespace`**：层级化路径，表示为字符串元组，如 `("users", "alice", "preferences")`。
- **`created_at`** / **`updated_at`**：时间戳，支持 ISO 8601 字符串自动解析。

使用 `__slots__` 减少内存占用。`__hash__` 基于 `(namespace, key)` 计算。

`SearchItem` 继承自 `Item`，增加 `score` 字段表示向量搜索的相似度分数。

## 命名空间设计

命名空间是 Store 最核心的组织概念，使用元组表示层级结构：

```python
("users",)                          # 所有用户
("users", "alice")                  # Alice 的数据
("users", "alice", "memories")      # Alice 的记忆
("global", "config")                # 全局配置
```

这种设计支持前缀匹配、通配符匹配和深度限制。`MatchCondition` 定义了匹配规则：

```python
# libs/checkpoint/langgraph/store/base/__init__.py

class MatchCondition(NamedTuple):
    match_type: NamespaceMatchType  # "prefix" | "suffix"
    path: NamespacePath             # tuple[str | Literal["*"], ...]
```

通配符 `"*"` 可以匹配任意元素。例如 `MatchCondition("prefix", ("users", "*", "memories"))` 匹配所有用户的 memories。

## 操作类型体系

Store 定义了四种操作，通过 `batch` 方法统一执行：

```python
# libs/checkpoint/langgraph/store/base/__init__.py

Op = GetOp | SearchOp | PutOp | ListNamespacesOp
```

**GetOp** 通过 `(namespace, key)` 精确获取：

```python
class GetOp(NamedTuple):
    namespace: tuple[str, ...]
    key: str
    refresh_ttl: bool = True
```

**SearchOp** 支持结构化过滤和语义搜索：

```python
class SearchOp(NamedTuple):
    namespace_prefix: tuple[str, ...]
    filter: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0
    query: str | None = None
```

`filter` 支持 MongoDB 风格操作符：`$eq`、`$ne`、`$gt`、`$gte`、`$lt`、`$lte`。

**PutOp** 写入或删除（`value=None` 表示删除）：

```python
class PutOp(NamedTuple):
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any] | None
    index: Literal[False] | list[str] | None = None
    ttl: float | None = None
```

`index` 控制向量索引：`None` 用默认配置，`False` 禁用，`list[str]` 指定字段路径。

## BaseStore 接口

```python
# libs/checkpoint/langgraph/store/base/__init__.py

class BaseStore(ABC):
    supports_ttl: bool = False

    @abstractmethod
    def batch(self, ops: Iterable[Op]) -> list[Result]: ...

    @abstractmethod
    async def abatch(self, ops: Iterable[Op]) -> list[Result]: ...
```

核心设计是**批量操作优先**——子类只需实现 `batch`/`abatch`，所有高层接口基于它们构建：

```python
def get(self, namespace, key, *, refresh_ttl=None) -> Item | None:
    return self.batch([GetOp(namespace, str(key), ...)])[0]

def search(self, namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0, ...) -> list[SearchItem]:
    return self.batch([SearchOp(namespace_prefix, filter, limit, offset, query, ...)])[0]

def put(self, namespace, key, value, index=None, *, ttl=...) -> None:
    self.batch([PutOp(namespace, str(key), value, index=index, ttl=...)])

def delete(self, namespace, key) -> None:
    self.batch([PutOp(namespace, str(key), None, ttl=None)])
```

这意味着子类只需优化一个 `batch` 方法就能同时优化所有操作。对于数据库后端，可以在一个事务中执行多个操作。

## InMemoryStore 实现

### 数据结构

```python
# libs/checkpoint/langgraph/store/memory/__init__.py

class InMemoryStore(BaseStore):
    __slots__ = ("_data", "_vectors", "index_config", "embeddings")

    def __init__(self, *, index: IndexConfig | None = None) -> None:
        self._data: dict[tuple[str, ...], dict[str, Item]] = defaultdict(dict)
        self._vectors: dict[tuple[str, ...], dict[str, dict[str, list[float]]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
```

`_data` 是二级字典 `namespace -> key -> Item`。`_vectors` 是三级字典 `namespace -> key -> field_path -> embedding`，存储向量索引。

### batch 执行流程

```python
# libs/checkpoint/langgraph/store/memory/__init__.py

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

1. `_prepare_ops` 遍历操作：GetOp 立即执行（直接字典查找），SearchOp 和 PutOp 分别收集。
2. 搜索阶段：生成 query embedding，执行批量相似度搜索。
3. 写入阶段：生成 embedding，应用写入。

### 搜索实现

`_filter_items` 按命名空间前缀和 filter 条件筛选候选项：

```python
# libs/checkpoint/langgraph/store/memory/__init__.py

def _filter_items(self, op: SearchOp) -> list[tuple[Item, list[list[float]]]]:
    filtered = []
    for namespace in self._data:
        if not (namespace[: len(op.namespace_prefix)] == op.namespace_prefix
                if len(namespace) >= len(op.namespace_prefix) else False):
            continue
        for key, item in self._data[namespace].items():
            if filter_func(item):
                if op.query and (embeddings := self._vectors[namespace].get(key)):
                    filtered.append((item, list(embeddings.values())))
                else:
                    filtered.append((item, []))
    return filtered
```

向量相似度使用余弦距离，优先用 NumPy 加速：

```python
# libs/checkpoint/langgraph/store/memory/__init__.py

def _cosine_similarity(X: list[float], Y: list[list[float]]) -> list[float]:
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
    # 纯 Python 回退...
```

搜索结果使用 **max pooling**：一个 Item 有多个向量时，取最高分。

### 过滤操作符

```python
# libs/checkpoint/langgraph/store/memory/__init__.py

def _apply_operator(value: Any, operator: str, op_value: Any) -> bool:
    if operator == "$eq":   return value == op_value
    elif operator == "$gt": return float(value) > float(op_value)
    elif operator == "$gte":return float(value) >= float(op_value)
    elif operator == "$lt": return float(value) < float(op_value)
    elif operator == "$lte":return float(value) <= float(op_value)
    elif operator == "$ne": return value != op_value
```

MongoDB 风格操作符，与 PostgresStore 的 JSONB 查询语义对齐。

### list_namespaces 与通配符

```python
# libs/checkpoint/langgraph/store/memory/__init__.py

def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
    all_namespaces = list(self._data.keys())
    namespaces = all_namespaces
    if op.match_conditions:
        namespaces = [ns for ns in namespaces
                      if all(_does_match(c, ns) for c in op.match_conditions)]
    if op.max_depth is not None:
        namespaces = sorted({ns[: op.max_depth] for ns in namespaces})
    else:
        namespaces = sorted(namespaces)
    return namespaces[op.offset : op.offset + op.limit]
```

`max_depth` 将命名空间截断到指定深度并去重。例如 `max_depth=1` 会将 `("users", "alice")` 和 `("users", "bob")` 都截断为 `("users",)`。

通配符匹配的实现：

```python
# libs/checkpoint/langgraph/store/memory/__init__.py

def _does_match(match_condition: MatchCondition, key: tuple[str, ...]) -> bool:
    if match_condition.match_type == "prefix":
        for k_elem, p_elem in zip(key, match_condition.path, strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
    elif match_condition.match_type == "suffix":
        for k_elem, p_elem in zip(reversed(key), reversed(match_condition.path), strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
```

## 向量搜索配置

通过 `IndexConfig` 启用语义搜索：

```python
# libs/checkpoint/langgraph/store/base/__init__.py

class IndexConfig(TypedDict, total=False):
    dims: int
    embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str
    fields: list[str] | None
```

使用示例：

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(index={
    "dims": 1536,
    "embed": "openai:text-embedding-3-small",
    "fields": ["text", "summary"],
})
store.put(("docs",), "doc1", {"text": "Python 教程", "summary": "入门指南"})
results = store.search(("docs",), query="编程入门")
```

`fields` 支持 JSON 路径语法：`"$"` 整个对象、`"field"` 顶层字段、`"parent.child"` 嵌套字段、`"array[*].field"` 数组元素。

## TTL 支持

```python
# libs/checkpoint/langgraph/store/base/__init__.py

class TTLConfig(TypedDict, total=False):
    refresh_on_read: bool
    default_ttl: float | None
    sweep_interval_minutes: int | None
```

TTL 需要子类声明 `supports_ttl = True`。InMemoryStore 不支持，PostgresStore 等生产实现通常支持。

## 本章要点

1. **Store 是跨线程的键值存储**，与 Checkpoint（绑定 thread_id）互补，适合用户画像、长期记忆等跨对话数据。
2. **namespace 使用元组表示层级路径**，支持前缀/后缀匹配和 `"*"` 通配符。
3. **BaseStore 批量操作优先**——子类只需实现 `batch`/`abatch`，`get`/`put`/`search`/`delete`/`list_namespaces` 自动基于它们构建。
4. **InMemoryStore** 使用 `defaultdict` 存储，可选向量搜索通过 `IndexConfig` 配置，余弦相似度计算优先用 NumPy。
5. **过滤器**采用 MongoDB 风格操作符（`$eq`、`$gt` 等），与 PostgresStore 的 JSONB 语义一致。
6. **Item** 是存储基本单元，使用 `__slots__` 优化内存，`(namespace, key)` 构成唯一寻址坐标。
