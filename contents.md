# 目录

## 第一部分：宏观认知

- [第 1 章　项目概览与设计哲学](/chapters/01-overview)
- [第 2 章　Monorepo 结构与模块依赖](/chapters/02-repo-structure)

## 第二部分：图的构建 API

- [第 3 章　StateGraph：声明式图构建 API](/chapters/03-state-graph)
- [第 4 章　State 与 Channel：六种状态通道](/chapters/04-state-channel)
- [第 5 章　节点、边与路由](/chapters/05-node-edge)
- [第 6 章　编译：从声明图到可执行 Pregel](/chapters/06-compile)

## 第三部分：Pregel 执行引擎

- [第 7 章　Pregel 总览与任务调度](/chapters/07-pregel-model)
- [第 8 章　PregelLoop：执行主循环](/chapters/08-pregel-loop)
- [第 9 章　Streaming：六种输出模式](/chapters/09-streaming)
- [第 10 章　Retry 策略、错误处理与节点 Cache](/chapters/10-retry-cache)

## 第四部分：持久化系统

- [第 11 章　Checkpoint 抽象层](/chapters/11-checkpoint-base)
- [第 12 章　三种 Checkpointer 实现 + Serde](/chapters/12-checkpointer-impl)
- [第 13 章　Store：跨线程持久化键值存储](/chapters/13-store)

## 第五部分：控制流高级特性

- [第 14 章　Interrupt 与 Human-in-the-Loop](/chapters/14-interrupt)
- [第 15 章　Command 与时间旅行](/chapters/15-command-time-travel)
- [第 16 章　子图与命名空间隔离](/chapters/16-subgraph)
- [第 17 章　Functional API：@entrypoint 与 @task](/chapters/17-functional-api)

## 第六部分：预构建组件与平台

- [第 18 章　预构建 Agent、RemoteGraph、CLI 与 SDK](/chapters/18-prebuilt-platform)

## 附录

- [附录 A：推荐阅读路径](/chapters/appendix-a-reading-path)
- [附录 B：核心类型速查](/chapters/appendix-b-type-reference)
- [附录 C：名词解释（Glossary）](/chapters/appendix-c-glossary)
