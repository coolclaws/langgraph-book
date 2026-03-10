# 第 17 章 Functional API：@entrypoint 与 @task

前面的章节中，我们深入分析了 `StateGraph` 这种声明式的图定义方式。LangGraph 还提供了一种函数式的替代方案 -- Functional API，通过 `@entrypoint` 和 `@task` 两个装饰器实现。本章将完整剖析 `langgraph/func/__init__.py` 的 575 行实现，理解 Functional API 如何在保持 Pregel 引擎能力的同时，提供更直观的编程模型。

## @task 装饰器

`@task` 是 Functional API 中定义可执行单元的方式。它将一个普通函数包装为一个可以被 LangGraph 引擎管理的 task，支持 retry、cache、以及 checkpoint 集成。

### _TaskFunction 包装类

```python
# libs/langgraph/langgraph/func/__init__.py
class _TaskFunction(Generic[P, T]):
    def __init__(
        self,
        func: Callable[P, Awaitable[T]] | Callable[P, T],
        *,
        retry_policy: Sequence[RetryPolicy],
        cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
        name: str | None = None,
    ) -> None:
        if name is not None:
            if hasattr(func, "__func__"):
                instance_method = functools.partial(func.__func__, func.__self__)
                instance_method.__name__ = name
                func = instance_method
            else:
                func.__name__ = name
        self.func = func
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        functools.update_wrapper(self, func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> SyncAsyncFuture[T]:
        return call(
            self.func,
            retry_policy=self.retry_policy,
            cache_policy=self.cache_policy,
            *args,
            **kwargs,
        )
```

核心设计点：

1. **`__call__` 返回 Future**：调用 `@task` 装饰的函数不会立即执行，而是返回一个 `SyncAsyncFuture[T]`。这使得并发执行成为可能。
2. **name 覆盖**：支持通过参数指定 task 名称，覆盖函数的原始名称。对实例方法做了特殊处理，通过 `functools.partial` 包装避免修改原始类方法。
3. **CachePolicy 与 RetryPolicy**：直接存储在 `_TaskFunction` 上，在调用时传递给引擎。

### task 装饰器的多重重载

```python
# libs/langgraph/langgraph/func/__init__.py
@overload
def task(
    __func_or_none__: None = None,
    *,
    name: str | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
) -> Callable[
    [Callable[P, Awaitable[T]] | Callable[P, T]],
    _TaskFunction[P, T],
]: ...

@overload
def task(__func_or_none__: Callable[P, Awaitable[T]]) -> _TaskFunction[P, T]: ...

@overload
def task(__func_or_none__: Callable[P, T]) -> _TaskFunction[P, T]: ...
```

三种 overload 支持三种使用方式：

```python
# 方式一：无参装饰器
@task
def add_one(a: int) -> int:
    return a + 1

# 方式二：带参装饰器
@task(retry_policy=RetryPolicy(max_attempts=3))
def risky_operation(data: str) -> str:
    return call_external_api(data)

# 方式三：异步函数
@task
async def async_add_one(a: int) -> int:
    return a + 1
```

实现的核心逻辑很简洁：

```python
# libs/langgraph/langgraph/func/__init__.py
def task(__func_or_none__=None, *, name=None, retry_policy=None, cache_policy=None):
    retry_policies: Sequence[RetryPolicy] = (
        ()
        if retry_policy is None
        else (retry_policy,)
        if isinstance(retry_policy, RetryPolicy)
        else retry_policy
    )

    def decorator(func):
        return _TaskFunction(
            func, retry_policy=retry_policies, cache_policy=cache_policy, name=name
        )

    if __func_or_none__ is not None:
        return decorator(__func_or_none__)

    return decorator
```

### call() 函数：task 的执行入口

`_TaskFunction.__call__` 调用的 `call()` 函数定义在 `_call.py` 中：

```python
# libs/langgraph/langgraph/pregel/_call.py
def call(
    func: Callable[P, Awaitable[T]] | Callable[P, T],
    *args: Any,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    **kwargs: Any,
) -> SyncAsyncFuture[T]:
    config = get_config()
    impl = config[CONF][CONFIG_KEY_CALL]
    fut = impl(
        func,
        (args, kwargs),
        retry_policy=retry_policy,
        cache_policy=cache_policy,
        callbacks=config["callbacks"],
    )
    return fut
```

`CONFIG_KEY_CALL` 是注入到 config 中的实际执行器。它负责：
1. 将函数和参数打包为一个 task
2. 通过引擎调度执行
3. 返回一个 Future，允许调用者决定何时获取结果

`SyncAsyncFuture` 同时支持同步和异步等待：

```python
# libs/langgraph/langgraph/pregel/_call.py
class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self) -> Generator[T, None, T]:
        yield cast(T, ...)
```

它继承了 `concurrent.futures.Future`（提供 `.result()` 方法），同时实现了 `__await__` 协议（支持 `await`）。

### task 的 Runnable 包装

在引擎内部，task 函数被包装为 `RunnableSeq`，最后会附加一个 `ChannelWrite` 来写入 RETURN channel：

```python
# libs/langgraph/langgraph/pregel/_call.py
def get_runnable_for_task(func: Callable[..., Any]) -> Runnable:
    # ...
    seq = RunnableSeq(
        run,
        ChannelWrite([ChannelWriteEntry(RETURN)]),
        name=name,
        trace_inputs=functools.partial(
            _explode_args_trace_inputs, inspect.signature(func)
        ),
    )
    return seq
```

RETURN channel 用于记录 task 的返回值，使得 checkpointer 可以在恢复时跳过已完成的 task。

## @entrypoint 装饰器

`@entrypoint` 定义工作流的入口点。它是一个类，而不是普通函数，这样设计是为了支持 `entrypoint.final` 属性。

### entrypoint 类结构

```python
# libs/langgraph/langgraph/func/__init__.py
class entrypoint(Generic[ContextT]):
    def __init__(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        cache: BaseCache | None = None,
        context_schema: type[ContextT] | None = None,
        cache_policy: CachePolicy | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.store = store
        self.cache = cache
        self.cache_policy = cache_policy
        self.retry_policy = retry_policy
        self.context_schema = context_schema
```

`entrypoint` 接受 checkpointer、store、cache 等参数 -- 这些与 `StateGraph.compile()` 的参数完全对应。

### entrypoint.final：分离返回值与持久化值

```python
# libs/langgraph/langgraph/func/__init__.py
@dataclass(**_DC_KWARGS)
class final(Generic[R, S]):
    value: R
    """Value to return. A value will always be returned even if it is None."""
    save: S
    """The value for the state for the next checkpoint."""
```

`entrypoint.final` 允许函数返回一个值给调用者，同时保存另一个值到 checkpoint。这在需要累积状态的场景中非常有用：

```python
@entrypoint(checkpointer=InMemorySaver())
def counter(increment: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    new_total = previous + increment
    # 返回 new_total 给调用者，保存 new_total 到 checkpoint
    return entrypoint.final(value=new_total, save=new_total)
```

### __call__：将函数转换为 Pregel 图

`entrypoint.__call__` 是最核心的方法，它将装饰的函数转换为一个完整的 Pregel 图：

```python
# libs/langgraph/langgraph/func/__init__.py
def __call__(self, func: Callable[..., Any]) -> Pregel:
    if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
        raise NotImplementedError(
            "Generators are not supported in the Functional API."
        )

    bound = get_runnable_for_entrypoint(func)
    stream_mode: StreamMode = "updates"

    sig = inspect.signature(func)
    first_parameter_name = next(iter(sig.parameters.keys()), None)
    if not first_parameter_name:
        raise ValueError("Entrypoint function must have at least one parameter")
    input_type = (
        sig.parameters[first_parameter_name].annotation
        if sig.parameters[first_parameter_name].annotation
        is not inspect.Signature.empty
        else Any
    )
```

然后定义两个辅助函数用于处理 `entrypoint.final`：

```python
    def _pluck_return_value(value: Any) -> Any:
        return value.value if isinstance(value, entrypoint.final) else value

    def _pluck_save_value(value: Any) -> Any:
        return value.save if isinstance(value, entrypoint.final) else value
```

最后构建 Pregel 图：

```python
    graph: Pregel = Pregel(
        nodes={
            func.__name__: PregelNode(
                bound=bound,
                triggers=[START],
                channels=START,
                writers=[
                    ChannelWrite(
                        [
                            ChannelWriteEntry(END, mapper=_pluck_return_value),
                            ChannelWriteEntry(PREVIOUS, mapper=_pluck_save_value),
                        ]
                    )
                ],
            )
        },
        channels={
            START: EphemeralValue(input_type),
            END: LastValue(output_type, END),
            PREVIOUS: LastValue(save_type, PREVIOUS),
        },
        input_channels=START,
        output_channels=END,
        stream_channels=END,
        stream_mode=stream_mode,
        stream_eager=True,
        checkpointer=self.checkpointer,
        store=self.store,
        cache=self.cache,
        cache_policy=self.cache_policy,
        retry_policy=self.retry_policy or (),
        context_schema=self.context_schema,
    )
    return graph
```

这段代码揭示了 Functional API 的本质：**它是 StateGraph 的语法糖**。一个 `@entrypoint` 装饰的函数被转换为一个只有一个节点的 Pregel 图，具有三个 channel：

| Channel | 类型 | 作用 |
|---------|------|------|
| `START` | `EphemeralValue` | 接收函数输入 |
| `END` | `LastValue` | 存储函数返回值 |
| `PREVIOUS` | `LastValue` | 存储 checkpoint 持久化值 |

节点通过 `ChannelWrite` 将函数返回值分别写入 `END` 和 `PREVIOUS` channel。如果返回的是 `entrypoint.final` 对象，则 `_pluck_return_value` 提取 `value` 写入 END，`_pluck_save_value` 提取 `save` 写入 PREVIOUS。

### get_runnable_for_entrypoint：函数到 Runnable 的转换

```python
# libs/langgraph/langgraph/pregel/_call.py
def get_runnable_for_entrypoint(func: Callable[..., Any]) -> Runnable:
    key = (func, False)
    if key in CACHE:
        return CACHE[key]
    else:
        if is_async_callable(func):
            run = RunnableCallable(
                None, func, name=func.__name__, trace=False, recurse=False
            )
        else:
            afunc = functools.update_wrapper(
                functools.partial(run_in_executor, None, func), func
            )
            run = RunnableCallable(
                func, afunc,
                name=func.__name__, trace=False, recurse=False,
            )
        if not _lookup_module_and_qualname(func):
            return run
        return CACHE.setdefault(key, run)
```

对于同步函数，LangGraph 会自动创建一个异步版本（通过 `run_in_executor`），使得 Pregel 引擎可以统一使用异步调度。`CACHE` 字典确保相同函数不会重复创建 Runnable。

## 与 StateGraph 的区别

### 声明式 vs 函数式

StateGraph 是声明式的：你先定义节点和边，然后编译。控制流由图的结构决定。

```python
# StateGraph 方式
builder = StateGraph(State)
builder.add_node("step1", fn1)
builder.add_node("step2", fn2)
builder.add_edge("step1", "step2")
graph = builder.compile()
```

Functional API 是函数式的：你写一个普通函数，控制流就是函数的控制流。

```python
# Functional API 方式
@entrypoint(checkpointer=InMemorySaver())
def workflow(input_data: str) -> str:
    result1 = step1_task(input_data).result()
    result2 = step2_task(result1).result()
    return result2
```

### 状态管理的差异

StateGraph 通过 channels 管理状态，每个节点读写 channel。状态在每一步都会被持久化。

Functional API 的状态就是函数的局部变量和 `previous` 参数。只有在函数返回时才会持久化（写入 PREVIOUS channel）。task 的结果通过 RETURN channel 被 checkpoint 记录，恢复时可以跳过已完成的 task。

### 路由逻辑的差异

StateGraph 使用 `add_conditional_edges` 和 `Command(goto=...)` 实现路由。Functional API 使用普通的 `if/else` 和函数调用。

## @task 的并发执行

Functional API 的一大优势是自然的并发支持。由于 `@task` 返回 Future，你可以轻松实现并行执行：

### 同步并发

```python
@task
def process_item(item: str) -> str:
    return heavy_computation(item)

@entrypoint()
def parallel_workflow(items: list[str]) -> list[str]:
    # 同时提交所有 task
    futures = [process_item(item) for item in items]
    # 收集结果
    results = [f.result() for f in futures]
    return results
```

### 异步并发（asyncio.gather 模式）

```python
@task
async def async_process(item: str) -> str:
    return await async_heavy_computation(item)

@entrypoint()
async def async_parallel(items: list[str]) -> list[str]:
    futures = [async_process(item) for item in items]
    return await asyncio.gather(*futures)
```

在异步模式下，`SyncAsyncFuture.__await__` 的实现使得 `asyncio.gather` 可以正确工作。引擎在底层调度这些 task 时，会利用 Pregel 的并行执行能力。

## Checkpoint 集成

`@entrypoint` 自动管理状态的持久化。当配置了 checkpointer 时：

1. **previous 参数**：函数可以声明一个 `previous` 参数来接收上次执行的保存值
2. **task 结果缓存**：已完成的 task 不会在恢复时重新执行
3. **interrupt 支持**：可以在 entrypoint 函数或 task 中使用 `interrupt()`

```python
@entrypoint(checkpointer=InMemorySaver())
def review_workflow(topic: str) -> dict:
    essay_future = compose_essay(topic)
    essay = essay_future.result()
    human_review = interrupt({
        "question": "Please provide a review",
        "essay": essay
    })
    return {
        "essay": essay,
        "review": human_review,
    }
```

恢复时，`compose_essay` 的结果从 checkpoint 中恢复（不会重新执行），而 `interrupt()` 返回用户提供的 resume 值。

## 本章要点

1. **Functional API 是 Pregel 的语法糖**：`@entrypoint` 创建一个单节点的 Pregel 图，包含 START、END、PREVIOUS 三个 channel。底层执行引擎与 StateGraph 完全相同。

2. **@task 返回 Future**：调用 task 不会立即执行，而是返回 `SyncAsyncFuture`。通过 `.result()` 或 `await` 获取结果。这种设计天然支持并发。

3. **entrypoint.final 分离返回值与持久化值**：`value` 返回给调用者，`save` 写入 checkpoint。下次执行时通过 `previous` 参数访问。

4. **函数签名约束**：entrypoint 函数必须有至少一个参数（作为输入）。`previous`、`config`、`runtime` 是可选的注入参数。不支持 generator 函数。

5. **task 的 Runnable 包装**：同步函数自动创建异步版本。task 执行结果写入 RETURN channel，被 checkpointer 记录，支持恢复时跳过已完成的 task。

6. **cache 与 identifier**：task 的缓存 key 基于函数的模块路径和限定名（`module.qualname`）。动态创建的函数无法被缓存。
