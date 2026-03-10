# 第 7 章 Pregel 总览与任务调度

在前面的章节中，我们逐层拆解了 Channel、StateGraph 和 Functional API。它们最终都编译为同一个运行时引擎 -- `Pregel`。本章将深入这颗引擎的核心：超步计算模型、任务准备算法以及并行执行机制。

> 源码路径约定
> - `libs/langgraph/langgraph/pregel/main.py` -- Pregel 类定义
> - `libs/langgraph/langgraph/pregel/_algo.py` -- 任务调度算法（约 1233 行）
> - `libs/langgraph/langgraph/pregel/_runner.py` -- 并行执行器

---

## 7.1 Bulk Synchronous Parallel 与超步

Pregel 类的名字直接致敬 Google 论文 *Pregel: A System for Large-Scale Graph Processing*。LangGraph 的执行模型完整复现了 BSP（Bulk Synchronous Parallel）范式：

1. **Plan（计划）** -- 确定本超步需要执行哪些 actor（节点）。
2. **Execute（执行）** -- 所有被选中的 actor 并发运行，其间 channel 更新对彼此不可见。
3. **Update（更新）** -- 将本超步所有 actor 的写入统一应用到 channel，进入下一超步。

这段描述直接来自 Pregel 类的 docstring：

```python
# libs/langgraph/langgraph/pregel/main.py

class Pregel(PregelProtocol[StateT, ContextT, InputT, OutputT], ...):
    """
    Each step consists of three phases:

    - **Plan**: Determine which **actors** to execute in this step.
    - **Execution**: Execute all selected **actors** in parallel,
        until all complete, or one fails, or a timeout is reached.
        During this phase, channel updates are invisible to actors
        until the next step.
    - **Update**: Update the channels with the values written by
        the **actors** in this step.

    Repeat until no **actors** are selected for execution, or a
    maximum number of steps is reached.
    """
```

在 LangGraph 语境中，一个"超步"（superstep）对应一次 `loop.tick()` + `loop.after_tick()` 的完整循环。step 计数器从 checkpoint 的 `metadata["step"] + 1` 开始，每完成一个超步递增一次。

---

## 7.2 Pregel 类：运行时骨架

`Pregel` 是整个框架的运行时核心。`StateGraph.compile()` 的最终产物就是一个 `Pregel` 实例。它的关键字段如下：

```python
# libs/langgraph/langgraph/pregel/main.py  (简化)

class Pregel(...):
    nodes: dict[str, PregelNode]              # 所有注册的节点
    channels: dict[str, BaseChannel | ManagedValueSpec]  # 所有 channel
    trigger_to_nodes: Mapping[str, Sequence[str]]  # channel -> 被触发的节点列表

    input_channels: str | Sequence[str]       # 输入 channel
    output_channels: str | Sequence[str]      # 输出 channel
    stream_channels: str | Sequence[str] | None  # 流式 channel

    stream_mode: StreamMode = "values"
    checkpointer: Checkpointer = None
    store: BaseStore | None = None
    cache: BaseCache | None = None
    retry_policy: Sequence[RetryPolicy] = ()
    cache_policy: CachePolicy | None = None
    step_timeout: float | None = None
    interrupt_after_nodes: All | Sequence[str]
    interrupt_before_nodes: All | Sequence[str]
```

### 7.2.1 trigger_to_nodes 映射

`trigger_to_nodes` 是一个重要的优化数据结构。它记录 "哪个 channel 的更新会触发哪些节点"。当 step N 完成时，`apply_writes` 返回 `updated_channels` 集合。下一步 `prepare_next_tasks` 不需要遍历所有节点，只需查表 `trigger_to_nodes[channel]` 即可快速确定候选节点。

### 7.2.2 TASKS channel

Pregel 构造函数中有一段隐藏的重要逻辑：

```python
# libs/langgraph/langgraph/pregel/main.py

def __init__(self, ...):
    ...
    self.channels[TASKS] = Topic(Send, accumulate=False)
```

`TASKS` 是一个特殊的 `Topic` channel，用于存放 `Send` 对象（动态 PUSH 任务）。每当 conditional edge 返回 `Send(node, arg)` 时，Send 被写入 `TASKS` channel，在下一个超步中被 `prepare_next_tasks` 消费并转化为可执行任务。

---

## 7.3 invoke 与 stream：执行入口

`invoke` 方法实际上是对 `stream` 的封装。它调用 `stream` 收集所有输出，然后返回最终值：

```python
# libs/langgraph/langgraph/pregel/main.py (invoke 方法，简化)

def invoke(
    self,
    input: InputT | Command | None,
    config: RunnableConfig | None = None,
    *,
    stream_mode: StreamMode = "values",
    ...
) -> dict[str, Any] | Any:
    latest: dict[str, Any] | Any = None
    chunks: list[dict[str, Any] | Any] = []
    for chunk in self.stream(input, config, stream_mode=..., ...):
        if stream_mode == "values":
            latest = chunk  # 保留最后一个值
        else:
            chunks.append(chunk)
    return latest  # 或 chunks
```

真正的核心逻辑在 `stream` 方法中。它创建 `SyncPregelLoop` 和 `PregelRunner`，然后进入主循环：

```python
# libs/langgraph/langgraph/pregel/main.py (stream 方法核心循环)

with SyncPregelLoop(
    input,
    stream=StreamProtocol(stream.put, stream_modes),
    config=config,
    checkpointer=checkpointer,
    nodes=self.nodes,
    specs=self.channels,
    ...
) as loop:
    runner = PregelRunner(
        submit=config[CONF].get(
            CONFIG_KEY_RUNNER_SUBMIT,
            weakref.WeakMethod(loop.submit),
        ),
        put_writes=weakref.WeakMethod(loop.put_writes),
        node_finished=config[CONF].get(CONFIG_KEY_NODE_FINISHED),
    )
    # BSP 主循环：每次 tick 是一个超步
    while loop.tick():
        for task in loop.match_cached_writes():
            loop.output_writes(task.id, task.writes, cached=True)
        for _ in runner.tick(
            [t for t in loop.tasks.values() if not t.writes],
            timeout=self.step_timeout,
            get_waiter=get_waiter,
            schedule_task=loop.accept_push,
        ):
            yield from _output(...)  # 产出流式输出
        loop.after_tick()
```

这里可以清晰地看到 BSP 模型的映射：
- `loop.tick()` 对应 **Plan** 阶段，调用 `prepare_next_tasks` 确定要执行的任务
- `runner.tick()` 对应 **Execution** 阶段，并发执行所有任务
- `loop.after_tick()` 对应 **Update** 阶段，调用 `apply_writes` 将写入应用到 channel

---

## 7.4 _algo.py 核心：任务调度的三层架构

`_algo.py` 是整个 Pregel 运行时中最复杂的文件（约 1233 行），包含三大核心功能：

1. `apply_writes` -- 将写入应用到 channel
2. `prepare_next_tasks` -- 准备下一超步的任务集合
3. `prepare_single_task` -- 构造单个可执行任务

### 7.4.1 apply_writes：超步间的状态转移

```python
# libs/langgraph/langgraph/pregel/_algo.py

def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Iterable[WritesProtocol],
    get_next_version: GetNextVersion | None,
    trigger_to_nodes: Mapping[str, Sequence[str]],
) -> set[str]:
```

`apply_writes` 在每个超步结束时被调用，执行以下步骤：

**Step 1: 排序任务，确保确定性**

```python
tasks = sorted(tasks, key=lambda t: task_path_str(t.path[:3]))
```

按 `path` 前三段排序。这对于依赖 reducer 的 channel 至关重要 -- 同一 channel 收到多个写入时，应用顺序必须一致。

**Step 2: 更新 versions_seen**

```python
for task in tasks:
    checkpoint["versions_seen"].setdefault(task.name, {}).update(
        {
            chan: checkpoint["channel_versions"][chan]
            for chan in task.triggers
            if chan in checkpoint["channel_versions"]
        }
    )
```

每个任务执行后，将其 trigger channel 的当前版本号记录到 `versions_seen` 中。这是 "channel 已经被消费" 的标记。

**Step 3: 消费 trigger channel**

```python
for chan in {chan for task in tasks for chan in task.triggers ...}:
    if channels[chan].consume() and next_version is not None:
        checkpoint["channel_versions"][chan] = next_version
```

对所有已触发的 channel 调用 `consume()`。对于 `EphemeralValue`，这会清空其值；对于 `LastValue`，则不做任何事（值保持）。

**Step 4: 分组写入并应用**

```python
pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
for task in tasks:
    for chan, val in task.writes:
        if chan in (NO_WRITES, PUSH, RESUME, INTERRUPT, RETURN, ERROR):
            pass  # 跳过特殊 channel
        elif chan in channels:
            pending_writes_by_channel[chan].append(val)

updated_channels: set[str] = set()
for chan, vals in pending_writes_by_channel.items():
    if channels[chan].update(vals) and next_version is not None:
        checkpoint["channel_versions"][chan] = next_version
        if channels[chan].is_available():
            updated_channels.add(chan)
```

写入被按 channel 分组，然后批量 `update`。如果某 channel 在更新后变为可用状态（`is_available()` 返回 True），它就被记入 `updated_channels`，用于触发下一超步的节点。

**Step 5: 通知未更新的 channel 并检测终止**

```python
# 通知未更新但仍可用的 channel 新超步的到来
if bump_step:
    for chan in channels:
        if channels[chan].is_available() and chan not in updated_channels:
            if channels[chan].update(EMPTY_SEQ) and next_version is not None:
                ...

# 如果没有任何更新的 channel 能触发新任务，调用 finish()
if bump_step and updated_channels.isdisjoint(trigger_to_nodes):
    for chan in channels:
        if channels[chan].finish() and next_version is not None:
            ...
```

---

## 7.5 prepare_next_tasks：两类任务的统一调度

```python
# libs/langgraph/langgraph/pregel/_algo.py

def prepare_next_tasks(
    checkpoint: Checkpoint,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    stop: int,
    *,
    for_execution: bool,
    store: BaseStore | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    manager: None | ParentRunManager | AsyncParentRunManager = None,
    trigger_to_nodes: Mapping[str, Sequence[str]] | None = None,
    updated_channels: set[str] | None = None,
    retry_policy: Sequence[RetryPolicy] = (),
    cache_policy: CachePolicy | None = None,
) -> dict[str, PregelTask] | dict[str, PregelExecutableTask]:
```

该函数返回一个 `{task_id: task}` 字典。任务分为两大类：

### PUSH 任务（Send 机制）

```python
# libs/langgraph/langgraph/pregel/_algo.py (prepare_next_tasks 内部)

tasks_channel = cast(Topic[Send] | None, channels.get(TASKS))
if tasks_channel and tasks_channel.is_available():
    for idx, _ in enumerate(tasks_channel.get()):
        if task := prepare_single_task(
            (PUSH, idx), None, ...
        ):
            tasks.append(task)
```

PUSH 任务来自 `TASKS` channel 中的 `Send` 对象。每个 Send 对应一个 `(PUSH, idx)` 路径，通过 `prepare_push_task_send` 处理。这是 map-reduce 模式的基础 -- 当 conditional edge 返回 `[Send("node_a", x), Send("node_a", y)]` 时，两个 Send 会在下一超步被并行执行。

### PULL 任务（常规触发）

```python
# libs/langgraph/langgraph/pregel/_algo.py (prepare_next_tasks 内部)

if updated_channels and trigger_to_nodes:
    triggered_nodes: set[str] = set()
    for channel in updated_channels:
        if node_ids := trigger_to_nodes.get(channel):
            triggered_nodes.update(node_ids)
    candidate_nodes: Iterable[str] = sorted(triggered_nodes)
elif not checkpoint["channel_versions"]:
    candidate_nodes = ()
else:
    candidate_nodes = processes.keys()

for name in candidate_nodes:
    if task := prepare_single_task(
        (PULL, name), None, ...
    ):
        tasks.append(task)

return {t.id: t for t in tasks}
```

PULL 任务是常规的节点触发。优化逻辑很清晰：如果知道哪些 channel 被更新了，就只检查这些 channel 能触发的节点（通过 `trigger_to_nodes` 映射），否则回退到检查所有节点。`sorted(triggered_nodes)` 保证了确定性的节点执行顺序。

---

## 7.6 prepare_single_task：构建单个可执行任务

```python
# libs/langgraph/langgraph/pregel/_algo.py

def prepare_single_task(
    task_path: tuple[Any, ...],
    task_id_checksum: str | None,
    *,
    checkpoint: Checkpoint,
    checkpoint_id_bytes: bytes,
    checkpoint_null_version: V | None,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    stop: int,
    for_execution: bool,
    store: BaseStore | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    manager: None | ParentRunManager | AsyncParentRunManager = None,
    input_cache: dict[INPUT_CACHE_KEY_TYPE, Any] | None = None,
    cache_policy: CachePolicy | None = None,
    retry_policy: Sequence[RetryPolicy] = (),
) -> None | PregelTask | PregelExecutableTask:
```

根据 `task_path[0]` 分派到三个子函数：

| task_path 形式 | 调用目标 | 说明 |
|---|---|---|
| `(PUSH, ..., Call)` | `prepare_push_task_functional` | Functional API 的 call 任务 |
| `(PUSH, idx)` | `prepare_push_task_send` | Send 机制的 PUSH 任务 |
| `(PULL, name)` | 内联处理 | 常规 PULL 触发 |

对于 PULL 任务，核心逻辑是检查该节点的 trigger channel 是否有更新：

```python
# libs/langgraph/langgraph/pregel/_algo.py (PULL 分支)

elif task_path[0] == PULL:
    name = cast(str, task_path[1])
    if name not in processes:
        return
    proc = processes[name]
    if _triggers(
        channels,
        checkpoint["channel_versions"],
        checkpoint["versions_seen"].get(name),
        checkpoint_null_version,
        proc,
    ):
        triggers = tuple(sorted(proc.triggers))
        checkpoint_ns = f"{parent_ns}{NS_SEP}{name}" if parent_ns else name
        task_id = task_id_func(
            checkpoint_id_bytes, checkpoint_ns,
            str(step), name, PULL, *triggers,
        )
```

---

## 7.7 _triggers：版本比较触发机制

PULL 任务能否执行，取决于 `_triggers` 函数的判定：

```python
# libs/langgraph/langgraph/pregel/_algo.py

def _triggers(
    channels: Mapping[str, BaseChannel],
    versions: ChannelVersions,
    seen: ChannelVersions | None,
    null_version: V,
    proc: PregelNode,
) -> bool:
    if seen is None:
        for chan in proc.triggers:
            if channels[chan].is_available():
                return True
    else:
        for chan in proc.triggers:
            if channels[chan].is_available() and versions.get(
                chan, null_version
            ) > seen.get(chan, null_version):
                return True
    return False
```

逻辑很清晰：如果节点从未执行过（`seen is None`），只要 trigger channel 有值就触发；否则，比较 channel 的当前版本号与节点上次看到的版本号，只有版本更新了才触发。

这就是 BSP 模型中 "channel 更新延迟可见" 的具体实现 -- 写入发生在 step N，版本号在 `apply_writes` 中递增，但只有在 step N+1 的 `prepare_next_tasks` 中，节点才会通过版本比较发现更新。

---

## 7.8 任务图构建：trigger -> node -> channel write 依赖链

从宏观来看，Pregel 的执行形成了一条清晰的依赖链：

```
channel update (step N)
    |
    v
trigger_to_nodes[channel]  -->  candidate nodes
    |
    v
_triggers() version check  -->  confirmed tasks
    |
    v
prepare_single_task()  -->  PregelExecutableTask
    |
    v
node execution (parallel)
    |
    v
task.writes (deque)
    |
    v
apply_writes()  -->  channel update (step N+1)
    |
    v
... (repeat)
```

每一环都有明确的数据流：

1. **Channel 版本变化** -- `apply_writes` 递增 `channel_versions`
2. **节点触发** -- `_triggers` 通过比较 `channel_versions` 与 `versions_seen` 判断
3. **任务构建** -- `prepare_single_task` 包装输入、config、写入缓冲区
4. **并行执行** -- `PregelRunner` 提交到线程池
5. **写入收集** -- 每个任务的 `writes` deque 收集所有输出
6. **状态更新** -- `apply_writes` 统一应用到 channel

---

## 7.9 PregelExecutableTask：可执行任务的结构

当 `for_execution=True` 时，`prepare_single_task` 返回 `PregelExecutableTask`：

```python
# libs/langgraph/langgraph/types.py

@dataclass(frozen=True)
class PregelExecutableTask:
    name: str                          # 节点名
    input: Any                         # 输入数据
    proc: Runnable                     # 要执行的 Runnable
    writes: deque[tuple[str, Any]]     # 写入缓冲区（deque，线程安全）
    config: RunnableConfig             # 运行配置
    triggers: Sequence[str]            # 触发 channel 列表
    retry_policy: Sequence[RetryPolicy]  # 重试策略
    cache_key: CacheKey | None         # 缓存键
    id: str                            # 任务 ID
    path: tuple[str | int | tuple, ...]  # 任务路径
    writers: Sequence[Runnable] = ()   # 后处理写入器
    subgraphs: Sequence[PregelProtocol] = ()  # 子图引用
```

`writes` 字段使用 `deque` 而非 `list`，这是刻意的选择：

```python
# libs/langgraph/langgraph/pregel/_algo.py
writes: deque[tuple[str, Any]] = deque()
# ...
CONFIG_KEY_SEND: writes.extend,  # deque.extend is thread-safe
```

`deque.extend` 在 CPython 中是原子操作（受 GIL 保护），因此多个线程可以安全地向同一个 writes deque 追加数据，无需额外加锁。

### 7.9.1 config 的注入

每个任务的 `config` 中注入了丰富的运行时信息：

```python
# libs/langgraph/langgraph/pregel/_algo.py (创建 PULL 任务的 config)

configurable={
    CONFIG_KEY_TASK_ID: task_id,
    CONFIG_KEY_SEND: writes.extend,       # 写入函数
    CONFIG_KEY_READ: partial(             # 本地读取函数
        local_read, scratchpad, channels, managed,
        PregelTaskWrites(task_path[:3], name, writes, triggers),
    ),
    CONFIG_KEY_CHECKPOINTER: checkpointer,
    CONFIG_KEY_CHECKPOINT_MAP: {...},
    CONFIG_KEY_CHECKPOINT_NS: task_checkpoint_ns,
    CONFIG_KEY_SCRATCHPAD: scratchpad,
    CONFIG_KEY_RUNTIME: runtime,
}
```

- `CONFIG_KEY_SEND` -- 节点内部调用 `send` 时实际是向 `writes` deque 追加
- `CONFIG_KEY_READ` -- 节点内部读取当前状态时调用 `local_read`
- `CONFIG_KEY_SCRATCHPAD` -- 存放 interrupt 计数器和 resume 值

---

## 7.10 local_read：节点内的状态读取

`local_read` 是注入到每个任务 config 中的读取函数。当节点在执行过程中需要读取最新状态时，它提供了两种模式：

```python
# libs/langgraph/langgraph/pregel/_algo.py

def local_read(
    scratchpad: PregelScratchpad,
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    task: WritesProtocol,
    select: list[str] | str,
    fresh: bool = False,
) -> dict[str, Any] | Any:
    updated: dict[str, list[Any]] = defaultdict(list)
    # 收集当前任务的写入
    for c, v in task.writes:
        if c in select:
            updated[c].append(v)
    if fresh:
        # 创建 channel 副本并应用当前任务的写入
        local_channels: dict[str, BaseChannel] = {}
        for k in channels:
            cc = channels[k].copy()
            cc.update(updated[k])
            local_channels[k] = cc
        values = read_channels(local_channels, select)
    else:
        values = read_channels(channels, select)
    return values
```

- `fresh=False` -- 直接从 channel 读取当前快照值
- `fresh=True` -- 创建 channel 的副本，将当前任务的写入应用上去，返回最新值

这个机制让节点在同一超步中可以看到自己的写入，但看不到其他并行节点的写入 -- 完美符合 BSP 模型的隔离语义。这就是条件边（如 `should_continue`）能够根据节点最新输出做路由判断的原因。

---

## 7.11 并行执行：PregelRunner 与 concurrent.futures

`PregelRunner` 负责在一个超步中并发执行所有任务：

```python
# libs/langgraph/langgraph/pregel/_runner.py

class PregelRunner:
    def __init__(
        self,
        *,
        submit: weakref.ref[Submit],
        put_writes: weakref.ref[Callable[[str, Sequence[tuple[str, Any]]], None]],
        use_astream: bool = False,
        node_finished: Callable[[str], None] | None = None,
    ) -> None:
        self.submit = submit
        self.put_writes = put_writes
        self.use_astream = use_astream
        self.node_finished = node_finished
```

### 7.11.1 tick 方法的单任务快速路径

`tick` 方法是同步执行的核心。对单任务有一个重要的快速路径优化 -- 直接在当前线程执行，省去线程池开销：

```python
# libs/langgraph/langgraph/pregel/_runner.py (tick 方法)

def tick(self, tasks, *, reraise=True, timeout=None, retry_policy=None,
         get_waiter=None, schedule_task):
    tasks = tuple(tasks)
    futures = FuturesDict(
        callback=weakref.WeakMethod(self.commit),
        event=threading.Event(),
        future_type=concurrent.futures.Future,
    )
    yield  # 将控制权交还调用方

    # 单任务快速路径：直接在当前线程执行
    if len(tasks) == 1 and timeout is None and get_waiter is None:
        t = tasks[0]
        try:
            run_with_retry(t, retry_policy, ...)
            self.commit(t, None)
        except Exception as exc:
            self.commit(t, exc)
        ...
        return

    # 多任务：提交到线程池并发执行
    for t in tasks:
        fut = self.submit()(run_with_retry, t, retry_policy, ...)
        futures[fut] = t
```

### 7.11.2 多任务的 wait 循环

对于多个任务，`tick` 使用 `concurrent.futures.wait` 等待任一任务完成：

```python
# libs/langgraph/langgraph/pregel/_runner.py

    while len(futures) > (1 if get_waiter is not None else 0):
        done, inflight = concurrent.futures.wait(
            futures,
            return_when=concurrent.futures.FIRST_COMPLETED,
            timeout=(max(0, end_time - time.monotonic()) if end_time else None),
        )
        if not done:
            break  # 超时
        for fut in done:
            task = futures.pop(fut)
        if _should_stop_others(done):
            break  # 有任务失败，停止所有其他任务
        yield  # 产出中间结果给调用方
```

这里的 `yield` 是流式输出的关键。`runner.tick()` 是一个 generator，它在每个任务完成时 yield，让外层循环有机会立即产出输出，而不是等所有任务完成。

### 7.11.3 FuturesDict：带回调的 Future 管理

`FuturesDict` 是一个特殊的字典，它在 future 完成时自动调用 `commit` 回调，并通过 `threading.Event` 协调同步：

```python
# libs/langgraph/langgraph/pregel/_runner.py

class FuturesDict(Generic[F, E], dict[F, PregelExecutableTask | None]):
    event: E
    callback: weakref.ref[Callable]
    counter: int
    done: set[F]
    lock: threading.Lock

    def on_done(self, task: PregelExecutableTask, fut: F) -> None:
        try:
            if cb := self.callback():
                cb(task, _exception(fut))
        finally:
            with self.lock:
                self.done.add(fut)
                self.counter -= 1
                if self.counter == 0 or _should_stop_others(self.done):
                    self.event.set()
```

### 7.11.4 commit：写入提交与错误处理

`commit` 方法在每个任务完成时被调用，根据任务的执行结果采取不同行为：

```python
# libs/langgraph/langgraph/pregel/_runner.py

def commit(self, task: PregelExecutableTask, exception: BaseException | None) -> None:
    if isinstance(exception, GraphInterrupt):
        # 保存中断信息到 checkpointer
        if exception.args[0]:
            writes = [(INTERRUPT, exception.args[0])]
            self.put_writes()(task.id, writes)
    elif exception:
        # 保存错误信息
        task.writes.append((ERROR, exception))
        self.put_writes()(task.id, task.writes)
    else:
        # 正常完成
        if not task.writes:
            task.writes.append((NO_WRITES, None))
        self.put_writes()(task.id, task.writes)
```

`_should_stop_others` 实现了 fail-fast 语义：当任何一个任务失败（非 `GraphBubbleUp`）时，所有其他正在执行的任务会被停止。

---

## 7.12 任务 ID 的确定性生成

每个任务都有一个确定性的唯一 ID，由 checkpoint ID、namespace、step、节点名和 trigger 信息哈希生成：

```python
# libs/langgraph/langgraph/pregel/_algo.py

def _xxhash_str(namespace: bytes, *parts: str | bytes) -> str:
    """Generate a UUID from the XXH3 hash of a namespace and str parts."""
    hex = xxh3_128_hexdigest(
        namespace + b"".join(p.encode() if isinstance(p, str) else p for p in parts)
    )
    return f"{hex[:8]}-{hex[8:12]}-{hex[12:16]}-{hex[16:20]}-{hex[20:32]}"
```

哈希函数有两个实现：新版 checkpoint（`v > 1`）使用 `_xxhash_str`，旧版使用 `_uuid5_str`。确定性 ID 确保了：

- 相同输入和状态下，任务 ID 完全一致
- checkpoint 重放时可以匹配 pending writes 到正确的任务
- 子图的任务 ID 包含父图的 namespace，避免冲突

---

## 7.13 PUSH 任务的两种形态

PUSH 任务在 `prepare_single_task` 中被路由到两个不同的准备函数。

### Send 机制（Graph API）

```python
# libs/langgraph/langgraph/pregel/_algo.py

def prepare_push_task_send(
    task_path: tuple[str, tuple],  # (PUSH, idx)
    ...
    processes: Mapping[str, PregelNode],
) -> PregelTask | PregelExecutableTask | None:
    idx = cast(int, task_path[1])
    sends: Sequence[Send] = channels[TASKS].get()
    packet = sends[idx]
    proc = processes[packet.node]
    ...
```

### Functional API 的 Call

```python
# libs/langgraph/langgraph/pregel/_algo.py

def prepare_push_task_functional(
    task_path: tuple[str, tuple, int, str, Call],
    # (PUSH, parent task path, idx of PUSH write, id of parent task, Call)
    ...
) -> PregelTask | PregelExecutableTask:
    call = task_path[-1]
    proc_ = get_runnable_for_task(call.func)
    name = proc_.name
    ...
```

两种 PUSH 任务在 task path 中附加了一个 bool 标志，影响 interrupt 的发射行为：

```python
# Send PUSH:  (..., False)  -- 应当发出 interrupt
# Call PUSH:  (..., True)   -- interrupt 由父任务处理
```

---

## 7.14 Scratchpad 与 input_cache

### PregelScratchpad

每个任务都有一个 `PregelScratchpad`，用于管理临时状态：

```python
# libs/langgraph/langgraph/pregel/_algo.py

return PregelScratchpad(
    step=step,
    stop=stop,
    call_counter=LazyAtomicCounter(),       # call() 调用计数
    interrupt_counter=LazyAtomicCounter(),   # interrupt() 调用计数
    resume=task_resume_write,               # resume 值列表
    get_null_resume=get_null_resume,        # 获取全局 resume 值
    subgraph_counter=LazyAtomicCounter(),   # 子图命名空间计数
)
```

`LazyAtomicCounter` 使用 `itertools.count(0).__next__` 实现原子递增，避免了 `+= 1` 的非线程安全问题。

### input_cache

`prepare_next_tasks` 内部维护一个 `input_cache` 字典。当多个节点订阅相同的 channel 组合时，`_proc_input` 函数会先查 cache，避免重复执行 `read_channels`。这在大量节点订阅同一 state channel 时显著减少开销。

---

## 7.15 完整执行时序

以一个两步计算为例，展示完整的执行时序：

```
Step 0 (Input):
  1. map_input() 将用户输入写入 input channel
  2. apply_writes() 更新 channel_versions
  3. _put_checkpoint(source="input")

Step 1 (First superstep):
  1. tick()
     - prepare_next_tasks(): _triggers 发现 input channel 已更新
     - 生成 PregelExecutableTask for node_a
  2. runner.tick(): 在线程池执行 node_a
     - node_a 读取 state，执行逻辑
     - node_a 的输出写入 writes deque
  3. after_tick()
     - apply_writes(): 将 node_a 的写入应用到 channels
     - _put_checkpoint(source="loop")

Step 2 (Second superstep):
  1. tick()
     - prepare_next_tasks(): _triggers 发现 node_a 写入的 channel 已更新
     - 生成 PregelExecutableTask for node_b
  2. runner.tick(): 在线程池执行 node_b
  3. after_tick()
     - apply_writes(): 无新 trigger -> 图终止

Output:
  - read_channels(output_keys) 返回最终结果
```

---

## 本章要点

1. **BSP 模型**：LangGraph 的 Pregel 引擎严格遵循 Bulk Synchronous Parallel 模型。每个超步分为 Plan/Execute/Update 三阶段，channel 更新在超步间才可见。

2. **任务分类**：PULL 任务由 channel 更新触发（常规边），PUSH 任务由 `Send` 对象或 `Call` 驱动（map-reduce 和 Functional API）。两者在 `prepare_next_tasks` 中统一调度。

3. **触发优化**：`trigger_to_nodes` 映射和 `updated_channels` 集合配合，避免每步遍历所有节点，将触发判断从 O(N) 优化到 O(K)（K = 受影响节点数）。

4. **确定性 ID**：任务 ID 由 checkpoint ID + namespace + step + 节点名 + 类型 + triggers 哈希生成，保证相同状态下重放的确定性。

5. **并行执行**：`PregelRunner` 通过 `concurrent.futures` 并行执行同一超步的所有任务。单任务走快速路径避免线程池开销。`deque.extend` 提供线程安全的写入收集。

6. **local_read 隔离**：节点执行期间通过 `local_read(fresh=True)` 可看到自己的写入，但看不到其他并行节点的写入，维护了 BSP 隔离语义。

7. **Scratchpad**：每个任务拥有独立的 `PregelScratchpad`，管理 interrupt/resume 计数器和子图命名空间，使用 `LazyAtomicCounter` 保证线程安全。
