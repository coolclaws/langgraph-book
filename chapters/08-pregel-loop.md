# 第 8 章 PregelLoop：执行主循环

上一章我们看到了 Pregel 的任务调度算法。本章深入执行主循环本身 -- `PregelLoop` 类及其同步/异步变体。这是 LangGraph 运行时中最关键的状态机，管理着从 checkpoint 加载到 INPUT -> EXECUTE -> OUTPUT 的完整生命周期。

> 源码路径
> - `libs/langgraph/langgraph/pregel/_loop.py` -- 约 1404 行
> - `libs/langgraph/langgraph/pregel/_runner.py` -- 任务执行器

---

## 8.1 PregelLoop 类层次

`_loop.py` 中定义了三个类，形成清晰的继承层次：

```
PregelLoop (基类)
    |
    +-- SyncPregelLoop (AbstractContextManager)
    |       同步执行，使用 ThreadPoolExecutor
    |
    +-- AsyncPregelLoop (AbstractAsyncContextManager)
            异步执行，使用 asyncio.Task
```

`PregelLoop` 是一个普通类（非 ABC），包含所有共享逻辑。两个子类只负责：
1. 实现 context manager 协议（`__enter__`/`__aenter__`）
2. 绑定同步或异步的 checkpointer 方法
3. 处理缓存的同步/异步差异

---

## 8.2 核心状态字段

```python
# libs/langgraph/langgraph/pregel/_loop.py

class PregelLoop:
    # 运行时配置
    config: RunnableConfig
    store: BaseStore | None
    stream: StreamProtocol | None
    step: int                  # 当前超步编号
    stop: int                  # 最大超步上限（step + recursion_limit + 1）

    # 输入
    input: Any | None

    # 检查点系统
    checkpointer: BaseCheckpointSaver | None
    checkpoint: Checkpoint
    checkpoint_config: RunnableConfig
    checkpoint_metadata: CheckpointMetadata
    checkpoint_pending_writes: list[PendingWrite]
    checkpoint_previous_versions: dict[str, str | float | int]
    checkpoint_id_saved: str

    # 执行状态
    status: Literal[
        "input", "pending", "done",
        "interrupt_before", "interrupt_after", "out_of_steps",
    ]
    tasks: dict[str, PregelExecutableTask]
    output: None | dict[str, Any] | Any
    updated_channels: set[str] | None

    # 图结构
    nodes: Mapping[str, PregelNode]
    specs: Mapping[str, BaseChannel | ManagedValueSpec]
    channels: Mapping[str, BaseChannel]
    managed: ManagedValueMapping
    trigger_to_nodes: Mapping[str, Sequence[str]]

    # 控制标志
    is_replaying: bool       # 是否正在重放 checkpoint
    is_nested: bool          # 是否是子图
    interrupt_before: All | Sequence[str]
    interrupt_after: All | Sequence[str]
    durability: Durability   # "sync" | "async" | "exit"
```

`status` 字段是整个状态机的核心。它只有六个可能的值，覆盖了执行生命周期的所有阶段。

---

## 8.3 SyncPregelLoop vs AsyncPregelLoop

两个子类的核心差异集中在三个方面：

### 8.3.1 Context Manager 入口

`SyncPregelLoop` 使用 `__enter__`/`__exit__`，`AsyncPregelLoop` 使用 `__aenter__`/`__aexit__`。两者在入口处执行相同的初始化逻辑：

```python
# libs/langgraph/langgraph/pregel/_loop.py (SyncPregelLoop.__enter__, 简化)

def __enter__(self) -> Self:
    # 1. 加载 checkpoint
    if not self.checkpointer:
        saved = None
    elif self.is_nested and (replay_state := ...):
        saved = replay_state.get_checkpoint(...)
    else:
        saved = self.checkpointer.get_tuple(self.checkpoint_config)

    # 2. 处理空 checkpoint
    if saved is None:
        saved = CheckpointTuple(
            self.checkpoint_config,
            empty_checkpoint(),
            {"step": -2},
            None, []
        )

    # 3. 恢复状态
    self.checkpoint = saved.checkpoint
    self.checkpoint_metadata = saved.metadata
    self.checkpoint_pending_writes = [
        (str(tid), k, v) for tid, k, v in saved.pending_writes
    ] if saved.pending_writes is not None else []

    # 4. 创建后台执行器
    self.submit = self.stack.enter_context(BackgroundExecutor(self.config))

    # 5. 从 checkpoint 恢复 channel 状态
    self.channels, self.managed = channels_from_checkpoint(
        self.specs, self.checkpoint
    )

    # 6. 注册 interrupt 抑制器
    self.stack.push(self._suppress_interrupt)

    # 7. 初始化步数和输入处理
    self.status = "input"
    self.step = self.checkpoint_metadata["step"] + 1
    self.stop = self.step + self.config["recursion_limit"] + 1

    # 8. 处理首轮输入或恢复
    self.updated_channels = self._first(
        input_keys=self.input_keys,
        updated_channels=...
    )
    return self
```

异步版本的关键差异：

```python
# libs/langgraph/langgraph/pregel/_loop.py (AsyncPregelLoop.__aenter__)

async def __aenter__(self) -> Self:
    # checkpoint 用 await 加载
    saved = await self.checkpointer.aget_tuple(self.checkpoint_config)
    # executor 用 async context manager
    self.submit = await self.stack.enter_async_context(
        AsyncBackgroundExecutor(self.config)
    )
    # 其余逻辑完全相同
    ...
```

### 8.3.2 Checkpointer 绑定

同步版本绑定同步方法，异步版本绑定异步方法：

```python
# SyncPregelLoop
self.checkpointer_put_writes = checkpointer.put_writes

# AsyncPregelLoop
self.checkpointer_put_writes = checkpointer.aput_writes
```

### 8.3.3 Checkpoint 顺序保证

两个版本都通过 `_checkpointer_put_after_previous` 保证 checkpoint 的顺序写入：

```python
# SyncPregelLoop
def _checkpointer_put_after_previous(self, prev, config, checkpoint, metadata, new_versions):
    try:
        if prev is not None:
            prev.result()  # 等待前一个 concurrent.futures.Future
    finally:
        cast(BaseCheckpointSaver, self.checkpointer).put(
            config, checkpoint, metadata, new_versions
        )

# AsyncPregelLoop
async def _checkpointer_put_after_previous(self, prev, config, checkpoint, metadata, new_versions):
    try:
        if prev is not None:
            await prev  # 等待前一个 asyncio.Task
    finally:
        await cast(BaseCheckpointSaver, self.checkpointer).aput(
            config, checkpoint, metadata, new_versions
        )
```

| 特性 | SyncPregelLoop | AsyncPregelLoop |
|---|---|---|
| Context manager | `__enter__` / `__exit__` | `__aenter__` / `__aexit__` |
| 后台执行器 | `BackgroundExecutor` (ThreadPool) | `AsyncBackgroundExecutor` (asyncio) |
| Checkpoint 获取 | `checkpointer.get_tuple()` | `await checkpointer.aget_tuple()` |
| Checkpoint 写入 | `checkpointer.put()` | `await checkpointer.aput()` |
| put_writes | `checkpointer.put_writes()` | `checkpointer.aput_writes()` |
| 顺序保证 | `concurrent.futures.Future.result()` | `await asyncio.Task` |
| 缓存查询 | `self.cache.get()` | `await self.cache.aget()` |
| ExitStack | `ExitStack` | `AsyncExitStack` |

---

## 8.4 主循环三阶段：INPUT -> EXECUTE -> OUTPUT

PregelLoop 的生命周期可以分为三个大阶段。以 `Pregel.stream()` 中的调用为例：

```python
# libs/langgraph/langgraph/pregel/main.py (stream 方法)

with SyncPregelLoop(input, ...) as loop:      # INPUT 阶段
    runner = PregelRunner(...)
    while loop.tick():                         # EXECUTE 阶段 (Plan)
        for _ in runner.tick(...):             # EXECUTE 阶段 (Execute)
            yield from _output(...)            # OUTPUT 阶段 (流式)
        loop.after_tick()                      # EXECUTE 阶段 (Update)
    # OUTPUT 阶段 (最终)
```

### 8.4.1 INPUT 阶段

发生在 `__enter__` / `__aenter__` 内部，由 `_first` 方法驱动。这个方法处理三种情况：

**情况 1: 正常输入**

```python
# libs/langgraph/langgraph/pregel/_loop.py (_first 方法)

elif input_writes := deque(map_input(input_keys, self.input)):
    # 丢弃上一次未完成的任务
    discard_tasks = prepare_next_tasks(
        self.checkpoint, self.checkpoint_pending_writes,
        self.nodes, self.channels, self.managed,
        self.config, self.step, self.stop,
        for_execution=True, store=None, checkpointer=None, manager=None,
        updated_channels=updated_channels,
    )
    # 应用输入写入
    updated_channels = apply_writes(
        self.checkpoint, self.channels,
        [*discard_tasks.values(), PregelTaskWrites((), INPUT, input_writes, [])],
        self.checkpointer_get_next_version, self.trigger_to_nodes,
    )
    # 保存输入 checkpoint
    self._put_checkpoint({"source": "input"})
```

这里有一个微妙但重要的细节：`discard_tasks` 的作用是让上次未完成的任务的 `versions_seen` 被更新，从而在下一轮不会被重复触发。

**情况 2: Resume（恢复执行）**

```python
if is_resuming:
    self.checkpoint["versions_seen"].setdefault(INTERRUPT, {})
    for k in self.channels:
        if k in self.checkpoint["channel_versions"]:
            version = self.checkpoint["channel_versions"][k]
            self.checkpoint["versions_seen"][INTERRUPT][k] = version
```

Resume 时将所有 channel 的当前版本标记为 INTERRUPT 已见过，防止 `should_interrupt` 再次触发中断。

**情况 3: Command 输入**

```python
if input_is_command:
    if (resume := cast(Command, self.input).resume) is not None:
        if resume_is_map := (
            isinstance(resume, dict)
            and all(is_xxh3_128_hexdigest(k) for k in resume)
        ):
            self.config[CONF][CONFIG_KEY_RESUME_MAP] = resume
        else:
            if len(self._pending_interrupts()) > 1:
                raise RuntimeError(
                    "When there are multiple pending interrupts, "
                    "you must specify the interrupt id when resuming."
                )

    writes: defaultdict[str, list[tuple[str, Any]]] = defaultdict(list)
    for tid, c, v in map_command(cmd=cast(Command, self.input)):
        writes[tid].append((c, v))
    for tid, ws in writes.items():
        self.put_writes(tid, ws)
```

`Command` 对象可以同时携带 `update`（状态更新）、`resume`（interrupt 恢复值）和 `goto`（跳转目标）。`map_command` 将 Command 拆解为 `(task_id, channel, value)` 三元组。

### 8.4.2 EXECUTE 阶段

由 `tick()` 和 `after_tick()` 驱动的超步循环。

**`tick()` -- Plan 子阶段**

```python
# libs/langgraph/langgraph/pregel/_loop.py

def tick(self) -> bool:
    """Execute a single iteration of the Pregel loop.
    Returns True if more iterations are needed.
    """

    # 检查是否超出步数限制
    if self.step > self.stop:
        self.status = "out_of_steps"
        return False

    # 准备下一批任务
    self.tasks = prepare_next_tasks(
        self.checkpoint, self.checkpoint_pending_writes,
        self.nodes, self.channels, self.managed,
        self.config, self.step, self.stop,
        for_execution=True,
        manager=self.manager,
        store=self.store,
        checkpointer=self.checkpointer,
        trigger_to_nodes=self.trigger_to_nodes,
        updated_channels=self.updated_channels,
        retry_policy=self.retry_policy,
        cache_policy=self.cache_policy,
    )

    # 如果没有任务，图执行完毕
    if not self.tasks:
        self.status = "done"
        return False

    # 匹配 pending writes 到任务
    if not self.is_replaying and self.checkpoint_pending_writes:
        self._match_writes(self.tasks)

    # 检查 interrupt_before
    if self.interrupt_before and should_interrupt(
        self.checkpoint, self.interrupt_before, self.tasks.values()
    ):
        self.status = "interrupt_before"
        raise GraphInterrupt()

    # 输出已有 writes 的缓存命中任务
    for task in self.tasks.values():
        if task.writes:
            self.output_writes(task.id, task.writes, cached=True)

    return True
```

`_match_writes` 将 checkpoint 中的 pending writes 匹配到对应的任务。这在 resume 场景中很重要 -- 之前已经完成的任务的写入可以直接恢复，无需重新执行：

```python
def _match_writes(self, tasks: Mapping[str, PregelExecutableTask]) -> None:
    for tid, k, v in self.checkpoint_pending_writes:
        if k in (ERROR, INTERRUPT, RESUME):
            continue
        if task := tasks.get(tid):
            task.writes.append((k, v))
```

**`after_tick()` -- Update 子阶段**

```python
# libs/langgraph/langgraph/pregel/_loop.py

def after_tick(self) -> None:
    # 1. 将所有任务的写入应用到 channel
    self.updated_channels = apply_writes(
        self.checkpoint, self.channels,
        self.tasks.values(),
        self.checkpointer_get_next_version,
        self.trigger_to_nodes,
    )

    # 2. 产出 values 流式输出
    if not self.updated_channels.isdisjoint(
        (self.output_keys,) if isinstance(self.output_keys, str) else self.output_keys
    ):
        self._emit("values", map_output_values, self.output_keys, writes, self.channels)

    # 3. 清除 pending writes
    self.checkpoint_pending_writes.clear()

    # 4. 关闭重放标志
    self.is_replaying = False

    # 5. 保存 checkpoint
    self._put_checkpoint({"source": "loop"})

    # 6. 检查 interrupt_after
    if self.interrupt_after and should_interrupt(
        self.checkpoint, self.interrupt_after, self.tasks.values()
    ):
        self.status = "interrupt_after"
        raise GraphInterrupt()

    # 7. 清除 resuming 标志
    self.config[CONF].pop(CONFIG_KEY_RESUMING, None)
```

关键的执行顺序：先 `apply_writes` 更新 channel，再 `_put_checkpoint` 保存状态，最后检查 `interrupt_after`。这确保了 interrupt 时 checkpoint 已经包含了当前超步的所有写入。

### 8.4.3 OUTPUT 阶段

输出通过 `_emit` 方法和 `StreamProtocol` 流式传输。最终输出在 `_suppress_interrupt` 中设置：

```python
# libs/langgraph/langgraph/pregel/_loop.py

def _suppress_interrupt(self, exc_type, exc_value, traceback):
    ...
    if exc_type is None:
        # 正常完成：读取最终 channel 值
        self.output = read_channels(self.channels, self.output_keys)
    elif isinstance(exc_value, GraphInterrupt) and not self.is_nested:
        # 中断但非嵌套图：抑制异常，仍然产出输出
        self.output = read_channels(self.channels, self.output_keys)
        return True  # 抑制异常
```

---

## 8.5 Checkpoint 写入时机

Checkpoint 的写入贯穿整个生命周期。理解写入时机对 debug 和性能调优至关重要。

### 8.5.1 写入时机一览

| 时间点 | metadata.source | 触发条件 |
|---|---|---|
| 输入处理后 | `"input"` | 有新输入写入 |
| 每个超步后 | `"loop"` | `after_tick()` 内 |
| 图退出时 | 复用当前 metadata | `durability="exit"` 时 |

### 8.5.2 _put_checkpoint 实现

```python
# libs/langgraph/langgraph/pregel/_loop.py

def _put_checkpoint(self, metadata: CheckpointMetadata) -> None:
    exiting = metadata is self.checkpoint_metadata
    if exiting and self.checkpoint["id"] == self.checkpoint_id_saved:
        return  # 已保存过

    if not exiting:
        metadata["step"] = self.step
        metadata["parents"] = self.config[CONF].get(CONFIG_KEY_CHECKPOINT_MAP, {})
        self.checkpoint_metadata = metadata

    do_checkpoint = self._checkpointer_put_after_previous is not None and (
        exiting or self.durability != "exit"
    )

    # 创建新 checkpoint
    self.checkpoint = create_checkpoint(
        self.checkpoint,
        self.channels if do_checkpoint else None,
        self.step,
        id=self.checkpoint["id"] if exiting else None,
        updated_channels=self.updated_channels,
    )

    if do_checkpoint:
        channel_versions = self.checkpoint["channel_versions"].copy()
        new_versions = get_new_channel_versions(
            self.checkpoint_previous_versions, channel_versions
        )
        self.checkpoint_previous_versions = channel_versions

        # 非阻塞保存，但保证顺序
        self._put_checkpoint_fut = self.submit(
            self._checkpointer_put_after_previous,
            getattr(self, "_put_checkpoint_fut", None),
            self.checkpoint_config,
            copy_checkpoint(self.checkpoint),
            self.checkpoint_metadata,
            new_versions,
        )

    if not exiting:
        self.step += 1
```

关键设计：

- **非阻塞保存**：checkpoint 写入通过 `submit` 提交到后台线程，不阻塞主循环
- **顺序保证**：`_checkpointer_put_after_previous` 接收前一个 Future，等待其完成后才写入新 checkpoint
- **durability 控制**：`durability="exit"` 时跳过中间 checkpoint
- **step 递增**：仅在非退出时递增 step

### 8.5.3 三种 durability 模式

| 模式 | checkpoint 写入时机 | put_writes 行为 | 适用场景 |
|---|---|---|---|
| `"sync"` | 每步同步写入 | 立即持久化 | 需要强一致性 |
| `"async"` | 每步异步写入 | 立即持久化 | 默认模式，平衡性能 |
| `"exit"` | 仅退出时写入 | 仅退出时持久化 | 短流程/高性能 |

在 `stream()` 方法中，`"sync"` 模式额外等待 checkpoint 完成：

```python
# libs/langgraph/langgraph/pregel/main.py  stream() 内
loop.after_tick()
if durability_ == "sync":
    loop._put_checkpoint_fut.result()
```

### 8.5.4 put_writes：增量写入持久化

除了完整的 checkpoint 保存，每个任务的写入也会增量持久化：

```python
# libs/langgraph/langgraph/pregel/_loop.py

def put_writes(self, task_id: str, writes: WritesT) -> None:
    if not writes:
        return

    # 去重特殊 channel 的写入
    if all(w[0] in WRITES_IDX_MAP for w in writes):
        writes = list({w[0]: w for w in writes}.values())

    # NULL_TASK_ID 的累积语义 vs 普通任务的替换语义
    if task_id == NULL_TASK_ID:
        self.checkpoint_pending_writes = [
            w for w in self.checkpoint_pending_writes
            if w[0] != task_id or w[1] not in WRITES_IDX_MAP
        ]
    else:
        self.checkpoint_pending_writes = [
            w for w in self.checkpoint_pending_writes if w[0] != task_id
        ]

    # 累积到 pending_writes
    self.checkpoint_pending_writes.extend((task_id, c, v) for c, v in writes)

    # 异步持久化到 checkpointer
    if self.durability != "exit" and self.checkpointer_put_writes is not None:
        self.submit(self.checkpointer_put_writes, config, writes_to_save, task_id)

    # 触发流式输出
    if hasattr(self, "tasks"):
        self.output_writes(task_id, writes)
```

这意味着即使图在超步中间崩溃，已完成任务的写入也不会丢失 -- 它们已经被持久化到 checkpointer。这是 LangGraph 可靠性的重要保障。

---

## 8.6 INTERRUPT / RESUME 的状态机转换

Interrupt/Resume 是 Human-in-the-Loop 模式的核心机制。它在 PregelLoop 中通过一系列精心设计的状态转换实现。

### 8.6.1 Interrupt 的触发

Interrupt 有两种触发方式：

**方式 1: interrupt_before / interrupt_after 配置**

```python
# libs/langgraph/langgraph/pregel/_loop.py (tick 方法)

# 执行前中断
if self.interrupt_before and should_interrupt(
    self.checkpoint, self.interrupt_before, self.tasks.values()
):
    self.status = "interrupt_before"
    raise GraphInterrupt()

# after_tick 中 - 执行后中断
if self.interrupt_after and should_interrupt(
    self.checkpoint, self.interrupt_after, self.tasks.values()
):
    self.status = "interrupt_after"
    raise GraphInterrupt()
```

**方式 2: 节点内部调用 interrupt()**

节点内部调用 `interrupt()` 会抛出 `GraphInterrupt` 异常，被 `PregelRunner.commit` 捕获并记录：

```python
# libs/langgraph/langgraph/pregel/_runner.py (commit 方法)

if isinstance(exception, GraphInterrupt):
    if exception.args[0]:
        writes = [(INTERRUPT, exception.args[0])]
        if resumes := [w for w in task.writes if w[0] == RESUME]:
            writes.extend(resumes)
        self.put_writes()(task.id, writes)
```

### 8.6.2 Interrupt 的抑制

对于顶层图（非嵌套），`GraphInterrupt` 不应传播给用户代码：

```python
# libs/langgraph/langgraph/pregel/_loop.py

def _suppress_interrupt(self, exc_type, exc_value, traceback):
    # durability="exit" 时需要在退出前保存 checkpoint
    if self.durability == "exit" and (
        not self.is_nested or exc_value is not None ...
    ):
        self._put_checkpoint(self.checkpoint_metadata)
        self._put_pending_writes()

    # 顶层图抑制 GraphInterrupt
    suppress = isinstance(exc_value, GraphInterrupt) and not self.is_nested
    if suppress:
        # 发射最后一个 "values" 事件
        if hasattr(self, "tasks") and self.checkpoint_pending_writes:
            if any(task.writes for task in self.tasks.values()):
                updated_channels = apply_writes(...)
                ...
        # 保存最终输出
        self.output = read_channels(self.channels, self.output_keys)
        return True  # 抑制异常
```

关键行为：子图中不抑制 -- `not self.is_nested`，子图的 interrupt 向上冒泡给父图。

### 8.6.3 Resume 的流程

当用户调用 `graph.invoke(Command(resume=value), config)` 时：

1. `_first` 检测到 `input_is_command` 且有 `resume` 值
2. resume 值通过 `map_command` 转为 `(task_id, RESUME, value)` 写入
3. `put_writes` 将 RESUME 持久化
4. `_first` 设置 `is_resuming=True`，更新 `versions_seen[INTERRUPT]`
5. `tick` 中 `prepare_next_tasks` 重新创建被中断的任务
6. 任务执行时，`_scratchpad` 从 pending writes 中提取 resume 值
7. `interrupt()` 函数检查 scratchpad.resume，找到值后直接返回而不中断

### 8.6.4 多 Interrupt 场景

一个节点内可以多次调用 `interrupt()`。Scratchpad 的 `interrupt_counter()` 为每次调用分配递增索引。Resume 时按索引匹配。

当多个任务同时 interrupt 时，必须使用 interrupt ID 的字典形式 resume：

```python
Command(resume={
    "interrupt_id_1": "answer_1",
    "interrupt_id_2": "answer_2",
})
```

`_pending_interrupts` 方法跟踪哪些 interrupt 尚未被 resume：

```python
# libs/langgraph/langgraph/pregel/_loop.py

def _pending_interrupts(self) -> set[str]:
    pending_interrupts: dict[str, str] = {}
    pending_resumes: set[str] = set()

    for task_id, write_type, value in self.checkpoint_pending_writes:
        if write_type == INTERRUPT:
            pending_interrupts[task_id] = value[0].id
        elif write_type == RESUME:
            pending_resumes.add(task_id)

    resumed_interrupt_ids = {
        pending_interrupts[task_id]
        for task_id in pending_resumes
        if task_id in pending_interrupts
    }

    hanging_interrupts = {
        interrupt_id
        for interrupt_id in pending_interrupts.values()
        if interrupt_id not in resumed_interrupt_ids
    }
    return hanging_interrupts
```

### 8.6.5 状态转换图

```
                    invoke(input)
                        |
                        v
    [input] --_first()--> [pending]
                            |
              +-------------+
              |             |
              v             v
         tick()=True   tick()=False
              |             |
              |       [done] / [out_of_steps]
              |
    +---------+---------+
    |                   |
    v                   v
[interrupt_before]    runner.tick()
    |                   |
    |              after_tick()
    |                   |
    |         +---------+---------+
    |         |                   |
    |         v                   v
    |    [interrupt_after]    loop continues
    |         |                   |
    v         v                   v
  raise GraphInterrupt      next tick()
```

---

## 8.7 Replay 与 Resuming 的区分

PregelLoop 维护了两个重要的布尔标志：

- **`is_replaying`**：从特定 checkpoint 恢复时为 True，表示当前步可能需要重放
- **`is_resuming`**：从中断恢复时为 True

两者的区别体现在 `_first` 方法中：

```python
# libs/langgraph/langgraph/pregel/_loop.py (_first 方法)

is_resuming = bool(self.checkpoint["channel_versions"]) and bool(
    configurable.get(
        CONFIG_KEY_RESUMING,
        self.input is None
        or input_is_command
        or (not self.is_nested and ...)
    )
)

# 重放时清除过时的 RESUME writes（除非正在 resume）
if self.is_replaying and not (
    (input_is_command and cast(Command, self.input).resume is not None)
    or configurable.get(CONFIG_KEY_RESUMING, False)
):
    self.checkpoint_pending_writes = [
        w for w in self.checkpoint_pending_writes if w[1] != RESUME
    ]
```

Replay 从特定 checkpoint 重放时，清除 RESUME writes 使 `interrupt()` 重新触发。Resume 则保留 RESUME writes 使 `interrupt()` 直接返回。

---

## 8.8 accept_push：动态任务调度

`accept_push` 方法在节点执行期间处理 `call()` 产生的动态 PUSH 任务：

```python
# libs/langgraph/langgraph/pregel/_loop.py

def accept_push(
    self, task: PregelExecutableTask, write_idx: int, call: Call | None = None
) -> PregelExecutableTask | None:
    """Accept a PUSH from a task, potentially returning a new task to start."""
    if pushed := cast(
        PregelExecutableTask | None,
        prepare_single_task(
            (PUSH, task.path, write_idx, task.id, call),
            None,
            checkpoint=self.checkpoint,
            ...
        ),
    ):
        # 产出 debug 输出
        self._emit("tasks", map_debug_tasks, [pushed])
        # 保存新任务
        self.tasks[pushed.id] = pushed
        # 匹配 pending writes
        if not self.is_replaying:
            self._match_writes({pushed.id: pushed})
        return pushed
```

这个方法被传递给 `PregelRunner.tick()` 作为 `schedule_task` 参数。当节点内部调用 `call()` 时，`accept_push` 动态创建并注册新任务。这允许在超步执行过程中动态扩展任务集合。

---

## 8.9 _emit 与 DuplexStream：流式输出

### 8.9.1 _emit 方法

```python
# libs/langgraph/langgraph/pregel/_loop.py

def _emit(self, mode, values, *args, **kwargs):
    if self.stream is None:
        return
    # "debug" 模式是 "checkpoints" + "tasks" 的包装
    debug_remap = mode in ("checkpoints", "tasks") and "debug" in self.stream.modes
    if mode not in self.stream.modes and not debug_remap:
        return
    for v in values(*args, **kwargs):
        if mode in self.stream.modes:
            self.stream((self.checkpoint_ns, mode, v))
        if debug_remap:
            self.stream((
                self.checkpoint_ns, "debug",
                {"step": ..., "timestamp": ..., "type": ..., "payload": v},
            ))
```

StreamChunk 的格式为 `(namespace, mode, data)` 三元组。

### 8.9.2 DuplexStream

当 PregelLoop 作为子图运行时，需要同时向自己的 stream 和父图的 stream 发送事件：

```python
# libs/langgraph/langgraph/pregel/_loop.py

def DuplexStream(*streams: StreamProtocol) -> StreamProtocol:
    def __call__(value: StreamChunk) -> None:
        for stream in streams:
            if value[1] in stream.modes:
                stream(value)
    return StreamProtocol(__call__, {mode for s in streams for mode in s.modes})
```

在 `__init__` 中自动检测并创建：

```python
if self.stream is not None and CONFIG_KEY_STREAM in config[CONF]:
    self.stream = DuplexStream(self.stream, config[CONF][CONFIG_KEY_STREAM])
```

---

## 8.10 完整生命周期示例

以一次带 interrupt 和 resume 的执行为例：

```
第一次调用: graph.invoke({"query": "hello"}, config)

__enter__()
  checkpointer.get_tuple() -> None
  saved = empty_checkpoint (step=-2)
  self.step = -1

_first()
  map_input("query", "hello") -> [("query", "hello")]
  apply_writes() -> updated_channels={"query"}
  _put_checkpoint(source="input")
  self.step = 0

tick() -> True
  prepare_next_tasks() -> {task_id: node_a_task}
  return True

runner.tick()
  node_a 执行: interrupt("请确认")
  GraphInterrupt 被 runner 捕获
  writes = [(INTERRUPT, [Interrupt(value="请确认")])]
  put_writes(task_id, writes)

after_tick()
  apply_writes() -> 没有数据 channel 更新
  _put_checkpoint(source="loop")
  self.step = 1

tick() -> False  (no new tasks)
  status = "done"

__exit__()
  _suppress_interrupt()
  output = {"query": "hello", "__interrupt__": (...)}
  return True  (suppress exception)


第二次调用: graph.invoke(Command(resume="yes"), config)

__enter__()
  checkpointer.get_tuple() -> saved (step=0, pending_writes=[INTERRUPT])
  self.step = 1

_first()
  input_is_command = True
  is_resuming = True
  map_command(Command(resume="yes")) -> [(task_id, RESUME, "yes")]
  put_writes(task_id, [(RESUME, "yes")])
  更新 versions_seen[INTERRUPT]

tick() -> True
  prepare_next_tasks() -> {task_id: node_a_task}
  _match_writes() -> 匹配 RESUME 写入
  return True

runner.tick()
  node_a 重新执行
  interrupt("请确认") -> scratchpad.resume[0] = "yes" -> 返回 "yes"
  node_a 继续执行，产出结果
  writes = [(channel, result)]

after_tick()
  apply_writes()
  _put_checkpoint(source="loop")

tick() -> False
  status = "done"

__exit__()
  output = final_result
```

---

## 本章要点

1. **三层类结构**：`PregelLoop` 基类封装核心逻辑，`SyncPregelLoop` 和 `AsyncPregelLoop` 分别处理同步/异步的 checkpoint 加载、executor 创建和 context manager 协议。

2. **三大阶段**：INPUT 阶段（`__enter__` + `_first`）处理输入或恢复；EXECUTE 阶段（`tick` + `runner.tick` + `after_tick`）执行超步循环；OUTPUT 阶段（`_emit` + `_suppress_interrupt`）产出最终结果。

3. **Checkpoint 写入时机**：输入处理后（`source="input"`）、每个超步后（`source="loop"`）、图退出时。`_checkpointer_put_after_previous` 保证写入顺序，`put_writes` 提供增量持久化。

4. **三种 durability**：`"async"` 异步写 checkpoint 不阻塞执行，`"sync"` 同步等待写入完成，`"exit"` 仅在退出时写入。

5. **INTERRUPT/RESUME 状态机**：interrupt 通过 `GraphInterrupt` 异常传播，顶层图通过 `_suppress_interrupt` 捕获并抑制。Resume 通过 `Command(resume=...)` 输入，scratchpad 按索引匹配多个 interrupt 的 resume 值。

6. **Replay vs Resume**：Replay 清除 RESUME writes 使 interrupt 重新触发；Resume 保留 RESUME writes 使 interrupt 直接返回。两者通过 `is_replaying` 和 `is_resuming` 标志区分。

7. **子图隔离**：`is_nested` 标志控制 interrupt 抑制（子图不抑制）、checkpoint namespace（子图附加父 namespace）和 resume 传播。`DuplexStream` 支持多层嵌套流式输出。

8. **动态任务调度**：`accept_push` 在节点执行期间处理 `call()` 产生的新任务，将其加入当前超步的任务集合并提交到线程池。
