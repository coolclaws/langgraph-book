---
layout: home

hero:
  name: "LangGraph 源码解析"
  text: "基于 Pregel 模型的 AI Agent 状态机框架"
  tagline: 从 StateGraph 声明到 Pregel 执行引擎，全面解读 LangGraph 的架构设计与实现细节
  actions:
    - theme: brand
      text: 开始阅读
      link: /chapters/01-overview
    - theme: alt
      text: 查看目录
      link: /contents
    - theme: alt
      text: GitHub
      link: https://github.com/coolclaws/langgraph-book

features:
  - icon:
      src: /icons/graph.svg
    title: 图计算引擎
    details: 深入 Pregel 执行模型，解析超步调度、Channel 通信、任务并发的完整实现，理解图驱动 Agent 的核心运行机制。

  - icon:
      src: /icons/state.svg
    title: 状态管理体系
    details: 剖析六种 Channel 类型、三种 State 定义方式、Checkpoint 持久化与 Store 跨线程存储的完整状态管理架构。

  - icon:
      src: /icons/control.svg
    title: 控制流与高级特性
    details: 覆盖 Interrupt、Command、时间旅行、子图隔离、Functional API，掌握 Human-in-the-Loop 的生产级实现。

  - icon:
      src: /icons/platform.svg
    title: 预构建组件与平台
    details: 解读 create_react_agent、ToolNode、RemoteGraph、CLI 与 SDK，从源码理解 LangGraph 的开箱即用能力。
---
