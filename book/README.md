# 手写数字识别 — 从零实现两层神经网络

> 写给零基础新手的深度学习入门小册子

---

## 阅读指南

本书分为**两大部分**：

- **第一部分：理论篇** — 从最基础的概念出发，一步步建立完整的神经网络知识体系。每个公式都有完整的推导过程，不跳步骤。
- **第二部分：项目篇** — 把理论"翻译"成代码，逐文件、逐行讲解项目实现，重点关注数据维度的变化。

**建议阅读顺序：** 按章节顺序阅读，不建议跳读。后面的章节会大量引用前面的推导结论。

---

## 目录

### 第一部分：理论篇

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| [第 1 章](chapter-01-neural-network.md) | 什么是神经网络 | 神经元、层的概念、网络结构 |
| [第 2 章](chapter-02-matrix.md) | 矩阵与维度 | 矩阵乘法、批量处理、形状推导 |
| [第 3 章](chapter-03-activation-functions.md) | 激活函数 | Sigmoid / ReLU / Softmax 及其完整求导 |
| [第 4 章](chapter-04-forward-propagation.md) | 前向传播 | 带维度标注的完整前向传播推导 |
| [第 5 章](chapter-05-loss-functions.md) | 损失函数 | 交叉熵损失的直觉与推导 |
| [第 6 章](chapter-06-backpropagation.md) | 反向传播 | 链式法则、梯度逐层推导 |
| [第 7 章](chapter-07-optimizers.md) | 参数更新与优化器 | SGD / Momentum / Adam |

### 第二部分：项目篇

| 章节 | 标题 | 对应文件 |
|------|------|----------|
| [第 8 章](chapter-08-project-overview.md) | 项目结构总览 | 所有文件 |
| [第 9 章](chapter-09-dataset.md) | 数据加载与处理 | `dataset.py` |
| [第 10 章](chapter-10-functions.md) | 激活与损失函数实现 | `functions.py` |
| [第 11 章](chapter-11-model.md) | 模型实现 | `model.py` |
| [第 12 章](chapter-12-optimizer-code.md) | 优化器实现 | `optimizer.py` |
| [第 13 章](chapter-13-trainer.md) | 训练器 | `trainer.py` |
| [第 14 章](chapter-14-main.md) | 完整训练流程 | `main.py` + `config.py` |

---

## 本书使用的符号约定

| 符号 | 含义 |
|------|------|
| `n` | 批量大小（batch size），即一次处理多少个样本 |
| `X` | 输入数据矩阵，shape = `(n, 784)` |
| `W1, b1` | 第一层的权重和偏置 |
| `W2, b2` | 第二层的权重和偏置 |
| `a1, z1` | 第一层的线性输出和激活输出 |
| `a2, y` | 第二层的线性输出和最终概率 |
| `t` | 真实标签，shape = `(n,)` |
| `L` | 损失值（标量） |
| `∂L/∂W` | 损失对参数 W 的偏导数（梯度） |
| `@` | 矩阵乘法（Python / numpy 语法） |
| `·` | 逐元素乘法（element-wise） |

---

*本书基于 Kaggle MNIST 数字识别数据集，使用纯 numpy 从零实现，不依赖 PyTorch / TensorFlow 等深度学习框架。*
