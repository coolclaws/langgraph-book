# 第 4 章 State 与 Channel：六种状态通道

上一章我们看到 `StateGraph` 在构建时会将 State schema 的每个字段解析为一个 channel。Channel 是 LangGraph 状态管理的核心抽象——节点不会直接读写共享内存，而是通过 channel 进行间接通信。不同的 channel 类型决定了状态的合并策略：覆盖、累加、归约、一次性消费等。本章将深入 `langgraph/channels/` 目录下的全部实现，解析六种 channel 的设计与适用场景。

## 4.1 BaseChannel 接口

所有 channel 的基类定义在 `langgraph/channels/base.py` 中：

```python
# langgraph/channels/base.py
class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    """Base class for all channels."""

    __slots__ = ("key", "typ")

    def __init__(self, typ: Any, key: str = "") -> None:
        self.typ = typ
        self.key = key
```

`BaseChannel` 使用三个泛型参数：

- **`Value`**：`get()` 返回的值类型，即下游节点读到的类型。
- **`Update`**：`update()` 接收的更新类型，即上游节点写入的类型。
- **`Checkpoint`**：checkpoint 存储的类型，用于持久化和恢复。

### 核心方法

BaseChannel 定义了五个关键方法，构成完整的生命周期：

```python
# langgraph/channels/base.py

@abstractmethod
def get(self) -> Value:
    """Return the current value of the channel.
    Raises EmptyChannelError if the channel is empty (never updated yet)."""

@abstractmethod
def update(self, values: Sequence[Update]) -> bool:
    """Update the channel's value with the given sequence of updates.
    The order of the updates in the sequence is arbitrary.
    This method is called by Pregel for all channels at the end of each step.
    Returns True if the channel was updated, False otherwise."""

def checkpoint(self) -> Checkpoint | Any:
    """Return a serializable representation of the channel's current state.
    Raises EmptyChannelError if the channel is empty."""
    try:
        return self.get()
    except EmptyChannelError:
        return MISSING

def consume(self) -> bool:
    """Notify the channel that a subscribed task ran.
    A channel can use this method to modify its state,
    preventing the value from being consumed again.
    Returns True if the channel was updated, False otherwise."""
    return False

def finish(self) -> bool:
    """Notify the channel that the Pregel run is finishing.
    A channel can use this method to modify its state, preventing finish.
    Returns True if the channel was updated, False otherwise."""
    return False
```

几个设计要点：

1. **`update()` 接收 `Sequence`**：因为在一个 super-step 中可能有多个并行节点同时写入同一个 channel，所有写入值被收集后一次性传入。
2. **`consume()` 和 `finish()`**：这两个 hook 默认是 no-op，只有特定 channel（如 `NamedBarrierValue`、`EphemeralValue`）会覆盖它们来实现"消费后清除"或"完成后触发"的语义。
3. **`from_checkpoint()`**：从 checkpoint 恢复 channel 状态，是持久化支持的基础。

此外还有 `from_checkpoint()` 用于从持久化数据恢复 channel 状态，`copy()` 用于创建 channel 副本（默认通过 checkpoint/from_checkpoint 实现），以及 `is_available()` 用于检查 channel 是否有值可读。

## 4.2 LastValue：覆盖语义

`LastValue` 是最简单也是最常用的 channel——它只保留最后写入的值，对应 State 中没有 reducer 注解的普通字段。

```python
# langgraph/channels/last_value.py
class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received, can receive at most one value per step."""

    __slots__ = ("value",)

    def __init__(self, typ: Any, key: str = "") -> None:
        super().__init__(typ, key)
        self.value = MISSING
```

关键行为在 `update()` 方法中：

```python
# langgraph/channels/last_value.py
def update(self, values: Sequence[Value]) -> bool:
    if len(values) == 0:
        return False
    if len(values) != 1:
        msg = create_error_message(
            message=f"At key '{self.key}': Can receive only one value per step. "
                    "Use an Annotated key to handle multiple values.",
            error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
        )
        raise InvalidUpdateError(msg)
    self.value = values[-1]
    return True
```

**核心约束**：每个 step 只允许接收**一个**值。如果两个并行节点同时写入同一个 `LastValue` channel，会抛出 `InvalidUpdateError`。这是一个重要的安全保障——它强制开发者在并行写入场景下使用 reducer。

```python
# langgraph/channels/last_value.py
def get(self) -> Value:
    if self.value is MISSING:
        raise EmptyChannelError()
    return self.value
```

`get()` 在 channel 从未被写入时抛出 `EmptyChannelError`，这让 Pregel 知道该 channel 尚不可用。

### 适用场景

- State 中的普通字段，如 `count: int`、`current_step: str`
- 每个 step 只由一个节点更新的字段

## 4.3 BinaryOperatorAggregate：reducer 语义

当字段使用 `Annotated[type, reducer_func]` 注解时，LangGraph 会为其创建 `BinaryOperatorAggregate` channel：

```python
# langgraph/channels/binop.py
class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the result of applying a binary operator to the current value
    and each new value."""

    __slots__ = ("value", "operator")

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]):
        super().__init__(typ)
        self.operator = operator
        typ = _strip_extras(typ)
        if typ in (collections.abc.Sequence, collections.abc.MutableSequence):
            typ = list
        if typ in (collections.abc.Set, collections.abc.MutableSet):
            typ = set
        if typ in (collections.abc.Mapping, collections.abc.MutableMapping):
            typ = dict
        try:
            self.value = typ()
        except Exception:
            self.value = MISSING
```

构造器尝试用类型的无参构造函数创建初始值（如 `list()` -> `[]`），并将 `Sequence`、`Set` 等抽象类型自动映射到 `list`、`set` 等具体类型。

`update()` 是核心——依次对所有传入值应用 reducer：

```python
# langgraph/channels/binop.py
def update(self, values: Sequence[Value]) -> bool:
    if not values:
        return False
    if self.value is MISSING:
        self.value = values[0]
        values = values[1:]
    seen_overwrite: bool = False
    for value in values:
        is_overwrite, overwrite_value = _get_overwrite(value)
        if is_overwrite:
            if seen_overwrite:
                msg = create_error_message(
                    message="Can receive only one Overwrite value per super-step.",
                    error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                )
                raise InvalidUpdateError(msg)
            self.value = overwrite_value
            seen_overwrite = True
            continue
        if not seen_overwrite:
            self.value = self.operator(self.value, value)
    return True
```

注意 `Overwrite` 的支持——通过返回 `Overwrite(value)` 对象，节点可以跳过 reducer 直接覆盖当前值，但每个 super-step 只允许一次。

### 适用场景

典型用法如 `messages: Annotated[list, operator.add]`（列表追加）、`total: Annotated[int, operator.add]`（数值累加）等。

## 4.4 Topic：追加列表

`Topic` 是一个发布-订阅式的 channel，值始终是一个列表：

```python
# langgraph/channels/topic.py
class Topic(
    Generic[Value],
    BaseChannel[Sequence[Value], Value | list[Value], list[Value]],
):
    """A configurable PubSub Topic.

    Args:
        typ: The type of the value stored in the channel.
        accumulate: Whether to accumulate values across steps.
            If False, the channel will be emptied after each step.
    """

    __slots__ = ("values", "accumulate")

    def __init__(self, typ: type[Value], accumulate: bool = False) -> None:
        super().__init__(typ)
        self.accumulate = accumulate
        self.values = list[Value]()
```

`accumulate` 参数控制关键行为——是否跨 step 保留历史值：

```python
# langgraph/channels/topic.py
def update(self, values: Sequence[Value | list[Value]]) -> bool:
    updated = False
    if not self.accumulate:
        updated = bool(self.values)
        self.values = list[Value]()
    if flat_values := tuple(_flatten(values)):
        updated = True
        self.values.extend(flat_values)
    return updated
```

当 `accumulate=False`（默认）时，每次 `update` 先清空旧值再添加新值——这意味着 `get()` 只返回当前 step 写入的值。当 `accumulate=True` 时，值会跨 step 累积。

### 适用场景

- 事件流收集（`accumulate=False`）：每个 step 产生的事件列表
- 日志累积（`accumulate=True`）：跨 step 收集所有日志

## 4.5 EphemeralValue：一次性通道

`EphemeralValue` 存储当前 step 写入的值，下一个 step 开始时自动清除：

```python
# langgraph/channels/ephemeral_value.py
class EphemeralValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the value received in the step immediately preceding, clears after."""

    __slots__ = ("value", "guard")

    def __init__(self, typ: Any, guard: bool = True) -> None:
        super().__init__(typ)
        self.guard = guard
        self.value = MISSING
```

`update()` 的行为与 `LastValue` 类似但有关键区别——当没有新值时**主动清除**旧值：

```python
# langgraph/channels/ephemeral_value.py
def update(self, values: Sequence[Value]) -> bool:
    if len(values) == 0:
        if self.value is not MISSING:
            self.value = MISSING
            return True
        else:
            return False
    if len(values) != 1 and self.guard:
        raise InvalidUpdateError(
            f"At key '{self.key}': EphemeralValue(guard=True) can receive only one "
            "value per step. Use guard=False if you want to store any one of "
            "multiple values."
        )
    self.value = values[-1]
    return True
```

`guard` 参数控制并行写入保护：`True`（默认）禁止多写入，`False` 取最后一个值。在 `compile()` 中，`START` channel 就是一个 `EphemeralValue`——用户输入写入后被第一个节点消费，之后自动清除。

### 适用场景

- 一次性触发信号、图的输入通道（`START`）、只在当前 step 可见的中间数据

## 4.6 NamedBarrierValue：同步屏障

`NamedBarrierValue` 实现了一个"等待所有命名值到齐"的屏障机制：

```python
# langgraph/channels/named_barrier_value.py
class NamedBarrierValue(Generic[Value], BaseChannel[Value, Value, set[Value]]):
    """A channel that waits until all named values are received before
    making the value available."""

    __slots__ = ("names", "seen")

    def __init__(self, typ: type[Value], names: set[Value]) -> None:
        super().__init__(typ)
        self.names = names
        self.seen: set[str] = set()
```

构造时传入一组预期的名称集合 `names`。每次 `update()` 时记录已到达的名称：

```python
# langgraph/channels/named_barrier_value.py
def update(self, values: Sequence[Value]) -> bool:
    updated = False
    for value in values:
        if value in self.names:
            if value not in self.seen:
                self.seen.add(value)
                updated = True
        else:
            raise InvalidUpdateError(
                f"At key '{self.key}': Value {value} not in {self.names}"
            )
    return updated
```

`get()` 只有在 `self.seen == self.names` 时才返回值，否则抛出 `EmptyChannelError`。`consume()` 在屏障触发后重置 `seen` 集合，使屏障可以再次使用。

这就是 `add_edge([node_a, node_b], target)` 多源同步边背后的实现原理——编译器为每组同步边创建一个 `NamedBarrierValue`，当所有前驱节点都完成执行并写入自己的名称后，屏障才会"放行"。

还有一个变体 `NamedBarrierValueAfterFinish`，它在所有名称到齐后不立即可用，而是额外等待 `finish()` 被调用后才变得可用，适用于需要两阶段同步的场景。

### 适用场景

- 多路并行后的汇聚（fan-in）
- `add_edge(["a", "b", "c"], "merge")` 的底层实现

## 4.7 UntrackedValue：不参与 checkpoint

`UntrackedValue` 在功能上类似 `LastValue`，但关键区别在于它**永远不被 checkpoint**：

```python
# langgraph/channels/untracked_value.py
class UntrackedValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received, never checkpointed."""

    __slots__ = ("value", "guard")

    def checkpoint(self) -> Value | Any:
        return MISSING

    def from_checkpoint(self, checkpoint: Value) -> Self:
        empty = self.__class__(self.typ, self.guard)
        empty.key = self.key
        return empty  # 注意：不恢复任何值
```

`checkpoint()` 永远返回 `MISSING`，`from_checkpoint()` 也不恢复旧值——这意味着该 channel 的值不会被持久化，从 checkpoint 恢复时总是空的，值只在当前运行的生命周期内有效。

`update()` 逻辑与 `LastValue` 类似，同样支持 `guard` 参数控制并行写入保护。与 `EphemeralValue` 不同的是，`UntrackedValue` 在没有新值时**不会**清除旧值——它在当前运行的整个生命周期内保持最后写入的值。

### 适用场景

- 大型临时数据（如文件内容），不希望占用 checkpoint 存储
- 运行时缓存，无需跨 checkpoint 保留
- 敏感信息，不希望被持久化

## 4.8 六种 Channel 对比总结

| Channel | 语义 | 并行写入 | 跨 step 保留 | Checkpoint |
|---------|------|---------|-------------|-----------|
| `LastValue` | 覆盖 | 禁止（抛异常） | 是 | 是 |
| `BinaryOperatorAggregate` | reducer 归约 | 支持（依次应用） | 是 | 是 |
| `Topic` | 追加列表 | 支持（展平合并） | 可配置 | 是 |
| `EphemeralValue` | 一次性 | 可配置 | 否（自动清除） | 是 |
| `NamedBarrierValue` | 同步屏障 | 支持 | 触发后重置 | 是 |
| `UntrackedValue` | 覆盖（不存盘） | 可配置 | 是（运行内） | 否 |

## 4.9 三种 State 定义方式

LangGraph 支持三种方式定义 State schema，每种方式最终都会被解析为 channel 字典。

### TypedDict（推荐）

```python
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    count: int                                    # -> LastValue
    messages: Annotated[list, operator.add]        # -> BinaryOperatorAggregate
```

这是最常用也是官方推荐的方式。`get_type_hints(State, include_extras=True)` 可以提取出包括 `Annotated` 元数据在内的完整类型信息。

### Pydantic BaseModel

```python
from pydantic import BaseModel

class State(BaseModel):
    count: int = 0
    messages: Annotated[list, operator.add] = []
```

Pydantic 模型同样支持 `get_type_hints`，channel 解析逻辑完全一致，还可享受 Pydantic 的数据校验能力。

### dataclass

```python
from dataclasses import dataclass, field

@dataclass
class State:
    count: int = 0
    messages: Annotated[list, operator.add] = field(default_factory=list)
```

三种方式在 channel 层面的行为完全相同——底层都通过 `get_type_hints(schema, include_extras=True)` 提取字段类型和注解信息。

对于极简场景也可以直接传入类型（如 `StateGraph(Annotated[list[AnyMessage], add_messages])`），此时 `_get_channels` 检测到没有 `__annotations__`，会创建一个 `__root__` channel——这正是旧版 `MessageGraph` 的实现方式。

## 4.10 Annotated 注解与 reducer 的绑定机制

channel 类型的自动推断是 LangGraph 开发体验的关键。核心逻辑在 `_get_channel` 函数及其辅助函数中：

```python
# langgraph/graph/state.py
def _get_channel(
    name: str, annotation: Any, *, allow_managed: bool = True
) -> BaseChannel | ManagedValueSpec:
    # 优先级 1：ManagedValue
    if manager := _is_field_managed_value(name, annotation):
        ...
    # 优先级 2：显式 Channel 注解
    elif channel := _is_field_channel(annotation):
        channel.key = name
        return channel
    # 优先级 3：reducer 函数
    elif channel := _is_field_binop(annotation):
        channel.key = name
        return channel
    # 优先级 4：默认 LastValue
    fallback: LastValue = LastValue(annotation)
    fallback.key = name
    return fallback
```

### 显式 Channel 注解

`_is_field_channel` 检查 `Annotated` 的元数据中是否有 `BaseChannel` 实例或子类：

```python
# langgraph/graph/state.py
def _is_field_channel(typ: type[Any]) -> BaseChannel | None:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        for item in meta:
            if isinstance(item, BaseChannel):
                return item
            elif isclass(item) and issubclass(item, BaseChannel):
                return item(typ.__origin__ if hasattr(typ, "__origin__") else typ)
    return None
```

例如 `events: Annotated[list[str], Topic(str, accumulate=True)]` 直接传 channel 实例，或 `temp: Annotated[str, EphemeralValue]` 传 channel 类（自动用字段类型实例化）。

### reducer 函数自动推断

`_is_field_binop` 检查 `Annotated` 最后一个元数据是否是接受两个位置参数的 callable：

```python
# langgraph/graph/state.py
def _is_field_binop(typ: type[Any]) -> BinaryOperatorAggregate | None:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and callable(meta[-1]):
            sig = signature(meta[-1])
            params = list(sig.parameters.values())
            if (
                sum(
                    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    for p in params
                )
                == 2
            ):
                return BinaryOperatorAggregate(typ, meta[-1])
            else:
                raise ValueError(
                    f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
                )
    return None
```

**关键细节**：reducer 必须恰好接受两个位置参数，否则抛出 `ValueError`。它检查的是 `meta[-1]`（最后一个元数据项），所以可以在 reducer 之前添加其他注解而不影响推断。

### 推断优先级总结

对于 `x: Annotated[T, meta1, meta2, ...]`：ManagedValue > BaseChannel 实例/子类 > 双参数 callable（BinaryOperatorAggregate） > 默认 LastValue。无注解字段 `x: T` 直接使用 `LastValue`。

## 本章要点

1. **BaseChannel 是所有状态通道的抽象基类**，定义了 `get()` / `update()` / `checkpoint()` / `consume()` / `finish()` 五个生命周期方法。`update()` 接收一个值序列（因为可能有并行写入），返回 bool 表示是否有更新。

2. **六种 Channel 实现覆盖了所有状态管理需求**：`LastValue`（覆盖）、`BinaryOperatorAggregate`（reducer 归约）、`Topic`（列表追加）、`EphemeralValue`（一次性消费）、`NamedBarrierValue`（同步屏障）、`UntrackedValue`（不持久化的覆盖）。

3. **`LastValue` 不允许并行写入**——这是有意为之的安全机制，强制开发者在并行场景下显式使用 reducer。

4. **`BinaryOperatorAggregate` 支持 `Overwrite`**——通过 `Overwrite(value)` 可以绕过 reducer 直接覆盖值，但每个 super-step 只允许一次。

5. **`EphemeralValue` 是 START channel 的实现**——`compile()` 中 `START` 绑定到 `EphemeralValue(input_schema)`，确保用户输入只在第一步可见。

6. **`NamedBarrierValue` 是多源同步边的实现**——`add_edge(["a", "b"], "c")` 编译时为每组同步边创建屏障 channel。

7. **Channel 类型由 `Annotated` 元数据自动推断**：无注解默认 `LastValue`；双参数 callable 创建 `BinaryOperatorAggregate`；也可显式传入 `BaseChannel` 实例。

8. **三种 State 定义方式**（TypedDict / Pydantic / dataclass）在 channel 层面行为一致，都通过 `get_type_hints(schema, include_extras=True)` 提取类型信息。
