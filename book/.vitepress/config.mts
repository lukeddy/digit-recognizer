import { defineConfig } from 'vitepress'

// GitHub Pages 部署时需要设置 base 为 /<repo-name>/
// Vercel 部署时 base 为 / (默认值)
// 通过环境变量控制，GitHub Actions 会注入 VITEPRESS_BASE
const base = (process.env.VITEPRESS_BASE as string) || '/'

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: '从零实现两层神经网络',
  description: '手写数字识别 — 写给零基础新手的深度学习入门小册子',

  // 开启 LaTeX 数学公式支持（需要 markdown-it-mathjax3）
  markdown: {
    math: true,
    // 代码块行号
    lineNumbers: true,
  },

  // 去掉 .html 后缀
  cleanUrls: true,

  themeConfig: {
    // 顶部导航
    nav: [
      { text: '首页', link: '/' },
      { text: '理论篇', link: '/chapter-01-neural-network' },
      { text: '项目篇', link: '/chapter-08-project-overview' },
      {
        text: '源码',
        link: 'https://github.com/leonyangdev/digit-recognizer',
      },
    ],

    // 左侧侧边栏
    sidebar: [
      {
        text: '导读',
        collapsed: false,
        items: [
          { text: '目录与简介', link: '/README' },
        ],
      },
      {
        text: '第一部分：理论篇',
        collapsed: false,
        items: [
          { text: '第 1 章：什么是神经网络', link: '/chapter-01-neural-network' },
          { text: '第 2 章：矩阵与维度', link: '/chapter-02-matrix' },
          { text: '第 3 章：激活函数', link: '/chapter-03-activation-functions' },
          { text: '第 4 章：前向传播', link: '/chapter-04-forward-propagation' },
          { text: '第 5 章：损失函数', link: '/chapter-05-loss-functions' },
          { text: '第 6 章：反向传播', link: '/chapter-06-backpropagation' },
          { text: '第 7 章：参数更新与优化器', link: '/chapter-07-optimizers' },
        ],
      },
      {
        text: '第二部分：项目篇',
        collapsed: false,
        items: [
          { text: '第 8 章：项目结构总览', link: '/chapter-08-project-overview' },
          { text: '第 9 章：数据加载与处理', link: '/chapter-09-dataset' },
          { text: '第 10 章：激活与损失函数实现', link: '/chapter-10-functions' },
          { text: '第 11 章：模型实现', link: '/chapter-11-model' },
          { text: '第 12 章：优化器实现', link: '/chapter-12-optimizer-code' },
          { text: '第 13 章：训练器', link: '/chapter-13-trainer' },
          { text: '第 14 章：完整训练流程', link: '/chapter-14-main' },
        ],
      },
    ],

    // 社交链接
    socialLinks: [
      {
        icon: 'github',
        link: 'https://github.com/leonyangdev/digit-recognizer',
      },
    ],

    // 页脚
    footer: {
      message: '基于 Kaggle MNIST 数据集，使用纯 numpy 从零实现',
      copyright: 'Copyright © 2026 · 以 MIT 协议开源',
    },

    // 右侧文章目录
    outline: {
      label: '本页目录',
      level: [2, 3],
    },

    // 上/下篇导航文字
    docFooter: {
      prev: '上一章',
      next: '下一章',
    },

    // 内置全文搜索
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索',
                buttonAriaLabel: '搜索文档',
              },
              modal: {
                noResultsText: '没有找到相关内容',
                resetButtonTitle: '清除搜索',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭',
                },
              },
            },
          },
        },
      },
    },

    // 编辑链接（可选）
    editLink: {
      pattern:
        'https://github.com/leonyangdev/digit-recognizer/edit/main/book/:path',
      text: '在 GitHub 上编辑此页',
    },

    // 最后更新时间
    lastUpdated: {
      text: '最后更新于',
    },
  },

  // 构建时注入 HTML head
  head: [
    ['meta', { name: 'theme-color', content: '#646cff' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: '从零实现两层神经网络' }],
    ['meta', {
      property: 'og:description',
      content: '手写数字识别 — 写给零基础新手的深度学习入门小册子',
    }],
  ],
})
