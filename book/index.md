---
layout: home

hero:
  name: "从零实现两层神经网络"
  text: "手写数字识别"
  tagline: 写给完全零基础新手的深度学习入门小册子 · 纯 numpy 实现 · 公式全部手推
  actions:
    - theme: brand
      text: 开始阅读 →
      link: /chapter-01-neural-network
    - theme: alt
      text: 查看目录
      link: /README
    - theme: alt
      text: GitHub 源码
      link: https://github.com/leonyangdev/digit-recognizer

features:
  - icon: 🧠
    title: 从零建立理论体系
    details: 从"什么是神经元"出发，逐步覆盖激活函数、前向传播、损失函数、反向传播、优化器，每个公式都附完整推导过程，不跳步骤。

  - icon: 📐
    title: 矩阵维度讲透彻
    details: 深度学习的难点之一是维度变化。本书对每一次矩阵运算都标注 shape，用具体数字手算验证，彻底消除维度困惑。

  - icon: 🔬
    title: 公式推导一五一十
    details: Sigmoid 导数、Softmax+交叉熵联合梯度、Adam 偏差修正……每个关键推导都逐步展开，看得见每一步的来龙去脉。

  - icon: 💻
    title: 理论与代码一一对照
    details: 第二部分逐文件讲解项目代码，每段代码旁边有对应的数学公式和维度分析，理论和代码相互印证。

  - icon: 📦
    title: 纯 numpy 实现
    details: 不依赖 PyTorch / TensorFlow，所有计算用 numpy 手写，让你真正理解深度学习框架在做什么，而不是调用黑盒。

  - icon: 🎯
    title: 两大部分，循序渐进
    details: 第一部分（理论篇）先建立完整知识框架；第二部分（项目篇）把理论翻译成代码。建议按顺序阅读，不建议跳读。
---

<div style="text-align: center; margin: 3rem 0 1rem">

## 本书覆盖的内容

</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; max-width: 800px; margin: 0 auto 4rem">

<div>

**第一部分：理论篇**

1. [什么是神经网络](/chapter-01-neural-network) — 神经元、层、两层网络
2. [矩阵与维度](/chapter-02-matrix) — 矩阵乘法、批量处理
3. [激活函数](/chapter-03-activation-functions) — Sigmoid / ReLU / Softmax 及求导
4. [前向传播](/chapter-04-forward-propagation) — 带维度标注的完整推导
5. [损失函数](/chapter-05-loss-functions) — 交叉熵的直觉与推导
6. [反向传播](/chapter-06-backpropagation) — 链式法则，梯度逐层推导
7. [优化器](/chapter-07-optimizers) — SGD / Momentum / Adam

</div>

<div>

**第二部分：项目篇**

8. [项目结构总览](/chapter-08-project-overview) — 文件职责与数据流
9. [数据加载与处理](/chapter-09-dataset) — MNIST、归一化
10. [激活与损失函数实现](/chapter-10-functions) — functions.py
11. [模型实现](/chapter-11-model) — model.py，反向传播代码
12. [优化器实现](/chapter-12-optimizer-code) — optimizer.py
13. [训练器](/chapter-13-trainer) — trainer.py，训练循环
14. [完整训练流程](/chapter-14-main) — main.py + config.py

</div>

</div>
