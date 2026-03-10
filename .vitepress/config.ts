import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'LangGraph 源码解析',
  description: '深入剖析 LangGraph —— 基于 Pregel 模型的 AI Agent 状态机框架',
  lang: 'zh-CN',

  base: '/',

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/langgraph-book/favicon.svg' }],
    ['meta', { name: 'theme-color', content: '#38b2ac' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'LangGraph 源码解析' }],
    ['meta', { property: 'og:description', content: '深入剖析 LangGraph —— 基于 Pregel 模型的 AI Agent 状态机框架' }],
  ],

  themeConfig: {
    logo: { src: '/logo.svg', alt: 'LangGraph' },

    nav: [
      { text: '开始阅读', link: '/chapters/01-overview' },
      { text: '目录', link: '/contents' },
      { text: 'GitHub', link: 'https://github.com/coolclaws/langgraph-book' },
    ],

    sidebar: [
      {
        text: '前言',
        items: [
          { text: '关于本书', link: '/' },
          { text: '完整目录', link: '/contents' },
        ],
      },
      {
        text: '第一部分：宏观认知',
        collapsed: false,
        items: [
          { text: '第 1 章　项目概览与设计哲学', link: '/chapters/01-overview' },
          { text: '第 2 章　Monorepo 结构与模块依赖', link: '/chapters/02-repo-structure' },
        ],
      },
      {
        text: '第二部分：图的构建 API',
        collapsed: false,
        items: [
          { text: '第 3 章　StateGraph：声明式图构建', link: '/chapters/03-state-graph' },
          { text: '第 4 章　State 与 Channel：六种状态通道', link: '/chapters/04-state-channel' },
          { text: '第 5 章　节点、边与路由', link: '/chapters/05-node-edge' },
          { text: '第 6 章　编译：从声明图到可执行 Pregel', link: '/chapters/06-compile' },
        ],
      },
      {
        text: '第三部分：Pregel 执行引擎',
        collapsed: false,
        items: [
          { text: '第 7 章　Pregel 总览与任务调度', link: '/chapters/07-pregel-model' },
          { text: '第 8 章　PregelLoop：执行主循环', link: '/chapters/08-pregel-loop' },
          { text: '第 9 章　Streaming：六种输出模式', link: '/chapters/09-streaming' },
          { text: '第 10 章　Retry、错误处理与节点 Cache', link: '/chapters/10-retry-cache' },
        ],
      },
      {
        text: '第四部分：持久化系统',
        collapsed: false,
        items: [
          { text: '第 11 章　Checkpoint 抽象层', link: '/chapters/11-checkpoint-base' },
          { text: '第 12 章　三种 Checkpointer 实现 + Serde', link: '/chapters/12-checkpointer-impl' },
          { text: '第 13 章　Store：跨线程键值存储', link: '/chapters/13-store' },
        ],
      },
      {
        text: '第五部分：控制流高级特性',
        collapsed: false,
        items: [
          { text: '第 14 章　Interrupt 与 Human-in-the-Loop', link: '/chapters/14-interrupt' },
          { text: '第 15 章　Command 与时间旅行', link: '/chapters/15-command-time-travel' },
          { text: '第 16 章　子图与命名空间隔离', link: '/chapters/16-subgraph' },
          { text: '第 17 章　Functional API', link: '/chapters/17-functional-api' },
        ],
      },
      {
        text: '第六部分：预构建组件与平台',
        collapsed: false,
        items: [
          { text: '第 18 章　预构建 Agent、CLI 与 SDK', link: '/chapters/18-prebuilt-platform' },
        ],
      },
      {
        text: '附录',
        collapsed: true,
        items: [
          { text: '附录 A：推荐阅读路径', link: '/chapters/appendix-a-reading-path' },
          { text: '附录 B：核心类型速查', link: '/chapters/appendix-b-type-reference' },
          { text: '附录 C：名词解释（Glossary）', link: '/chapters/appendix-c-glossary' },
        ],
      },
    ],

    outline: {
      level: [2, 3],
      label: '本页目录',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/coolclaws/langgraph-book' },
    ],

    footer: {
      message: '基于 MIT 协议发布',
      copyright: 'Copyright © 2025-present',
    },

    search: {
      provider: 'local',
    },
  },

  markdown: {
    lineNumbers: true,
  },
})
