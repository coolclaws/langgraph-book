# LangGraph 源码解析

> 深入剖析 LangGraph —— 基于 Pregel 模型的 AI Agent 状态机框架

## 关于本书

[LangGraph](https://github.com/langchain-ai/langgraph) 是一个低层次的 Agent 编排框架，受 Google Pregel 论文启发，以图计算模型管理有状态的 AI Agent。它不是高层抽象工具，而是为需要精细控制 Agent 行为的开发者提供的基础设施。

本书从源码层面系统梳理 LangGraph 的架构设计与实现细节，适合希望：

- 理解图驱动 Agent 编排框架内部原理的开发者
- 学习 Pregel 计算模型在 AI Agent 场景中应用的工程师
- 希望基于 LangGraph 进行深度定制或贡献代码的参与者
- 对状态管理、Checkpoint 持久化、Human-in-the-Loop 感兴趣的技术人员

## 目录

详见 [CONTENTS.md](./contents.md)

全书共 **18 章 + 3 附录**，分六个部分：

| 部分 | 章节 | 核心议题 |
|------|------|---------|
| 第一部分：宏观认知 | Ch 1–2 | 设计哲学、Monorepo 结构 |
| 第二部分：图的构建 API | Ch 3–6 | StateGraph、Channel、节点/边、编译 |
| 第三部分：Pregel 执行引擎 | Ch 7–10 | 超步调度、执行循环、Streaming、Retry |
| 第四部分：持久化系统 | Ch 11–13 | Checkpoint、Checkpointer、Store |
| 第五部分：控制流高级特性 | Ch 14–17 | Interrupt、Command、子图、Functional API |
| 第六部分：预构建组件与平台 | Ch 18 | ReAct Agent、CLI、SDK |

## 在线阅读

[https://coolclaws.github.io/langgraph-book/](https://coolclaws.github.io/langgraph-book/)

## License

本书内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证。
LangGraph 项目本身采用 MIT 许可证。
