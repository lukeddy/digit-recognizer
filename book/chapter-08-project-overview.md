# 第 8 章：项目结构总览

> 理论篇结束，进入项目篇。这章先给出全局视图，帮你在读具体代码前建立地图。

## 8.1 文件结构

```
digit_recognizer/
├── config.py        ← 所有超参数的"控制台"
├── dataset.py       ← 数据加载与预处理
├── functions.py     ← 激活函数、损失函数
├── gradient.py      ← 数值梯度（调试用）
├── model.py         ← 神经网络模型（核心）
├── optimizer.py     ← SGD / Momentum / Adam
├── trainer.py       ← 训练循环
├── visualizer.py    ← 绘制训练曲线
├── main.py          ← 程序入口，串联所有模块
└── data/
    └── train.csv    ← MNIST 数据集（Kaggle 格式）
```

---

## 8.2 数据流：从文件到预测

```
train.csv
    ↓ dataset.py: 读取、划分、归一化
x_train (29400, 784)  +  y_train (29400,)
x_test  (12600, 784)  +  y_test  (12600,)
    ↓
    ┌─── 每次训练取 batch_size=100 个样本 ───┐
    │  x_batch (100, 784)  t_batch (100,)    │
    │      ↓  model.py: 前向传播             │
    │  y_pred (100, 10)                      │
    │      ↓  functions.py: 损失计算          │
    │  loss (标量)                           │
    │      ↓  model.py: 反向传播             │
    │  grads: {W1:(784,50), b1:(50,),        │
    │          W2:(50,10),  b2:(10,)}        │
    │      ↓  optimizer.py: 参数更新         │
    │  params 更新                           │
    └──────────────────────────────────────┘
         ↓ 重复 num_epochs × iters_per_epoch 次
    history: train_loss, train_acc, test_acc
         ↓ visualizer.py
    训练曲线图
```

---

## 8.3 模块依赖关系

```
main.py
  ├── config.py       （超参数）
  ├── dataset.py      → config.py
  ├── model.py        → functions.py, config.py
  ├── optimizer.py    （独立）
  ├── trainer.py      → model.py, optimizer.py, config.py
  └── visualizer.py   （独立）
```

---

## 8.4 各模块职责一览

| 文件 | 职责 | 关键对象 |
|------|------|----------|
| `config.py` | 集中管理所有超参数，避免"魔法数字" | `TRAIN_CONFIG`, `MODEL_CONFIG` |
| `dataset.py` | 读 CSV，归一化，划分训练/测试集 | `load_mnist()` |
| `functions.py` | 激活函数和损失函数的纯函数实现 | `sigmoid`, `softmax`, `cross_entropy_error` |
| `gradient.py` | 数值梯度（中心差分），用于验证反向传播 | `numerical_gradient()` |
| `model.py` | 两层神经网络，前向传播 + 反向传播 | `TwoLayerNet` |
| `optimizer.py` | 三种优化器策略 | `SGD`, `Momentum`, `Adam` |
| `trainer.py` | 训练循环，记录 loss 和准确率 | `Trainer`, `TrainHistory` |
| `visualizer.py` | 绘制训练曲线图 | `plot_training_results()` |
| `main.py` | 把所有模块串联起来 | `main()` |

---

> **接下来**，我们逐个深入每个文件。

---

[← 第 7 章](chapter-07-optimizers.md) | [返回目录](README.md) | [第 9 章：数据处理 →](chapter-09-dataset.md)
