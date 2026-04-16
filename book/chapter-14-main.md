# 第 14 章：完整训练流程（main.py + config.py）

> 最后一章。把所有模块串联起来，看清楚整个程序的执行顺序，以及如何通过修改配置来做实验。

## 14.1 config.py：一切的控制台

`config.py` 集中管理所有超参数：

```python
MODEL_CONFIG = {
    'input_size': 784,
    'hidden_size': 50,
    'output_size': 10,
    'weight_init_std': 0.01,
}

TRAIN_CONFIG = {
    'learning_rate': 0.001,   # Adam 推荐 0.001
    'batch_size': 100,
    'num_epochs': 10,
    'grad_method': 'backprop',  # 反向传播（快）
    'optimizer': 'adam',        # 最稳定的优化器
}
```

**为什么要有 config.py？**

没有 config.py 时，超参数散落在各处：

```python
# model.py
self.params['W1'] = np.random.randn(784, 50) * 0.01  # 0.01 是魔法数字

# trainer.py
for i in range(10):    # 10 是魔法数字
    batch = x[np.random.choice(len(x), 100)]  # 100 是魔法数字
```

这些"魔法数字"很难管理——想改 batch_size，要找遍所有文件。有了 config.py，改一处，全局生效。

---

## 14.2 main.py：程序主入口

```python
def main():
    # 1. 加载数据
    x_train, x_test, y_train, y_test = load_mnist()
    # x_train: (29400, 784)  y_train: (29400,)
    # x_test:  (12600, 784)  y_test:  (12600,)
    describe_dataset(x_train, x_test, y_train, y_test)

    # 2. 初始化模型（随机参数）
    model = TwoLayerNet()
    # model.params['W1']: (784, 50)  随机小值
    # model.params['b1']: (50,)      全零
    # model.params['W2']: (50, 10)   随机小值
    # model.params['b2']: (10,)      全零

    # 3. 初始化优化器
    optimizer = build_optimizer(
        optimizer_name=TRAIN_CONFIG['optimizer'],     # 'adam'
        learning_rate=TRAIN_CONFIG['learning_rate'],  # 0.001
    )

    # 4. 训练
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        x_train=x_train, y_train=y_train,
        x_test=x_test,   y_test=y_test,
    )
    history = trainer.train()

    # 5. 可视化
    plot_training_results(history)

if __name__ == '__main__':
    main()
```

---

## 14.3 执行过程全景

```
python main.py
  │
  ├─ load_mnist()
  │    ├─ 读取 train.csv (42000行)
  │    ├─ 分离 X(42000,784) 和 y(42000,)
  │    ├─ train_test_split → (29400,784) + (12600,784)
  │    └─ MinMaxScaler 归一化 → [0,1]
  │
  ├─ TwoLayerNet()
  │    ├─ W1 = randn(784,50) * 0.01
  │    ├─ b1 = zeros(50)
  │    ├─ W2 = randn(50,10) * 0.01
  │    └─ b2 = zeros(10)
  │
  ├─ Adam(lr=0.001)
  │    └─ 初始化 m={}, v={}, t=0
  │
  ├─ trainer.train()  ← 核心，执行 2940 次迭代
  │    │
  │    ├─ 迭代 0:
  │    │    ├─ 随机取 batch: x(100,784), t(100,)
  │    │    ├─ 前向传播: y(100,10)
  │    │    ├─ 反向传播: grads{W1,b1,W2,b2}
  │    │    ├─ Adam.step(): 更新所有参数
  │    │    ├─ 记录 loss
  │    │    └─ 评估准确率（iteration=0，是294的倍数？0%0=0✓）
  │    │         用全量 x_train(29400,784) 和 x_test(12600,784) 评估
  │    │
  │    ├─ 迭代 1-293: 同上，不评估准确率
  │    │
  │    ├─ 迭代 294: Epoch 1 结束，评估准确率
  │    │
  │    ├─ ... 重复 ...
  │    │
  │    └─ 迭代 2939: 最后一次迭代
  │         ↓
  │    最终评估：打印最终准确率
  │
  └─ plot_training_results(history)
       ├─ 绘制 loss 曲线（2940 个点）
       └─ 绘制准确率曲线（11 个点：初始 + 10 个 Epoch）
```

---

## 14.4 实验配置指南

想做不同的实验？只需修改 `config.py`：

### 实验1：比较优化器

```python
# 分别运行三次，比较训练曲线
TRAIN_CONFIG = {
    'optimizer': 'sgd',        # 实验1
    'learning_rate': 0.1,
    ...
}

TRAIN_CONFIG = {
    'optimizer': 'momentum',   # 实验2
    'learning_rate': 0.01,
    ...
}

TRAIN_CONFIG = {
    'optimizer': 'adam',       # 实验3（当前配置）
    'learning_rate': 0.001,
    ...
}
```

### 实验2：验证反向传播的正确性

```python
TRAIN_CONFIG = {
    'grad_method': 'numerical',  # 用数值梯度（慢！但可以验证）
    ...
}
```

把 numerical 和 backprop 算出的梯度做比较，若差距很小（< 1e-5），说明反向传播实现正确。

### 实验3：改变网络大小

```python
MODEL_CONFIG = {
    'hidden_size': 100,   # 更大的隐藏层（更强表达力，更慢）
    # 或
    'hidden_size': 20,    # 更小的隐藏层（欠拟合风险）
}
```

### 实验4：更多训练轮次

```python
TRAIN_CONFIG = {
    'num_epochs': 20,   # 训练更久
}
```

---

## 14.5 结果解读

典型的训练输出（使用 backprop + adam）：

```
Epoch  1/10  train_acc=0.6123  test_acc=0.6089
Epoch  2/10  train_acc=0.8234  test_acc=0.8198
Epoch  3/10  train_acc=0.8756  test_acc=0.8712
...
Epoch 10/10  train_acc=0.9456  test_acc=0.9398

最终训练集准确率: 0.9456 (94.56%)
最终测试集准确率: 0.9398 (93.98%)
```

**如何判断训练是否正常？**

| 现象 | 可能原因 | 处理方式 |
|------|---------|---------|
| loss 不下降 | 学习率太小 | 增大 lr |
| loss 剧烈波动 | 学习率太大 | 减小 lr |
| train_acc 高，test_acc 低（差距 > 5%）| 过拟合 | 减小 hidden_size，增加数据 |
| train_acc 和 test_acc 都低 | 欠拟合 | 增大 hidden_size，增加 epochs |
| loss 变成 NaN | 数值溢出 | 减小 lr 或 weight_init_std |

---

## 14.6 整本书的知识图谱

```
理论篇                          项目篇
─────────────────────────────────────────────────────────
                                config.py
                                    ↓
第9章         dataset.py ←─── DATA_CONFIG
  ↓
MNIST数据           x_train(29400,784)
                    y_train(29400,)
                         ↓
第3章         functions.py ←── sigmoid, softmax, cross_entropy
第4-5章
第6章                model.py ←─── MODEL_CONFIG
                  TwoLayerNet
                  ├─ forward()      # 第4章：前向传播
                  ├─ loss()         # 第5章：交叉熵
                  └─ _backprop()    # 第6章：反向传播
                         ↓
第7章         optimizer.py ←── TRAIN_CONFIG['optimizer']
                  Adam/SGD/Momentum
                  └─ step()
                         ↓
                  trainer.py ←── TRAIN_CONFIG
                  Trainer.train()
                  ├─ mini-batch采样
                  ├─ 调用 model.gradient()
                  ├─ 调用 optimizer.step()
                  └─ 记录 history
                         ↓
                  visualizer.py
                  plot_training_results(history)
```

---

## 14.7 下一步学习建议

完成这个项目后，可以继续探索：

1. **加更多隐藏层** → 深层网络（Deep Neural Networks）
2. **用 ReLU 替换 Sigmoid** → 缓解梯度消失，训练更深的网络
3. **正则化（Dropout, L2）** → 防止过拟合
4. **卷积神经网络（CNN）** → 专门处理图像，精度可达 99%+
5. **用 PyTorch 重写** → 自动微分，GPU 加速（MPS on Mac）

---

## 14.8 恭喜！

你已经从零实现了一个完整的两层神经网络，理解了：

- 神经网络的结构（输入层、隐藏层、输出层）
- 激活函数的作用及其导数推导（Sigmoid、ReLU、Softmax）
- 前向传播的完整数据流和维度变化
- 损失函数（交叉熵）的设计原理
- 反向传播的完整推导（链式法则 → 具体公式 → 矩阵维度）
- 三种优化器（SGD、Momentum、Adam）的原理与实现

这是深度学习最核心的内容。更复杂的模型（Transformer、CNN、GAN）在结构上更复杂，但训练的基本原理是完全一致的。

---

[← 第 13 章](chapter-13-trainer.md) | [返回目录](README.md)
