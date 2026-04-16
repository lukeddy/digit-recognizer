# 第 13 章：训练器（trainer.py）

> 训练循环：把模型、优化器、数据组合在一起，执行完整的训练流程。

## 13.1 两个类的职责

```python
class TrainHistory:   # 数据容器：存储训练过程中的指标
    train_loss_list   # 每次迭代的 loss
    train_acc_list    # 每个 Epoch 的训练准确率
    test_acc_list     # 每个 Epoch 的测试准确率

class Trainer:        # 训练引擎：执行训练循环
    train()           # 返回 TrainHistory
```

---

## 13.2 几个重要概念

在理解代码之前，先弄清楚训练中常用的术语：

### Epoch（轮次）

**一个 Epoch = 模型看完训练集中所有样本一遍**。

本项目训练集有 29,400 个样本，每次取 batch_size=100 个，所以：

```
一个 Epoch 需要的迭代次数 = 29400 / 100 = 294 次迭代
```

训练 10 个 Epoch = 模型总共经历 29,400 × 10 次样本（有些重复）。

### Iteration（迭代）

**一次迭代 = 一次前向传播 + 反向传播 + 参数更新**，使用一个 mini-batch。

```
总迭代次数 = iters_per_epoch × num_epochs
           = 294 × 10 = 2,940 次
```

### Mini-batch（小批量）

每次不用所有训练数据，而是随机取一小批（100个样本）来估算梯度。

**为什么不用全量数据？**
- 全量数据（29,400个）计算一次梯度太慢
- Mini-batch 提供梯度的"噪声估计"，这种随机性反而帮助跳出局部极小值
- 可以实现多次更新，加速收敛

---

## 13.3 Trainer 初始化

```python
class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        x_train, y_train,
        x_test,  y_test,
        batch_size=100,
        num_epochs=10,
        grad_method='backprop',
        print_loss_every=1,
    ):
        ...
        n = x_train.shape[0]   # 训练集总样本数：29400
        self.iters_per_epoch = int(np.ceil(n / batch_size))   # ceil(29400/100) = 294
        self.total_iters     = self.iters_per_epoch * num_epochs  # 294 × 10 = 2940
```

**`np.ceil` vs `int`**

`np.ceil(29400 / 100) = 294`（向上取整），即使样本数不能被 batch_size 整除也没事——最后一批可以少于 100 个样本（代码里用随机采样，所以实际上不存在"最后一批"问题）。

---

## 13.4 train 方法：训练主循环

```python
def train(self):
    history = TrainHistory()
    n = self.x_train.shape[0]   # 29400

    self._print_train_header()

    for iteration in range(self.total_iters):   # 共 2940 次
```

### ① Mini-batch 随机采样

```python
        batch_idx = np.random.choice(n, self.batch_size)
        # n=29400, batch_size=100
        # batch_idx: 从 0~29399 中随机选 100 个索引（允许重复）
        # shape: (100,)

        x_batch = self.x_train[batch_idx]   # shape: (100, 784)
        t_batch = self.y_train[batch_idx]   # shape: (100,)
```

**`np.random.choice(n, batch_size)` 详解：**

```python
np.random.choice(29400, 100)
# 从 [0, 1, 2, ..., 29399] 中随机选 100 个数，允许重复（有放回抽样）
# 返回: array([1234, 5678, 9012, ...])  shape: (100,)
```

**有放回抽样的含义：** 同一个样本可能在一个 batch 里出现多次。这对训练影响不大，但简化了代码（无需管理"每轮覆盖所有样本"的逻辑）。

**`x_train[batch_idx]` 的含义：**

```python
x_train.shape = (29400, 784)
batch_idx = [1234, 5678, 9012, ..., 8765]   # 100 个随机索引

x_train[batch_idx]
# 取出第 1234 行、第 5678 行 ... 第 8765 行
# 结果 shape: (100, 784)
```

### ② 计算梯度

```python
        grads = self.model.gradient(x_batch, t_batch, method=self.grad_method)
        # grad_method='backprop' → 调用 _backprop_gradient
        # grads: {'W1': (784,50), 'b1': (50,), 'W2': (50,10), 'b2': (10,)}
```

### ③ 更新参数

```python
        self.optimizer.step(self.model.params, grads)
        # 把模型的参数字典和梯度字典都传给优化器
        # 优化器原地修改 params 中的数组
```

注意：`self.model.params` 是字典，传给优化器后优化器直接修改其中的 numpy 数组（原地修改）。这样模型的权重就被更新了，不需要把更新后的 params 再赋值回去。

### ④ 记录损失

```python
        loss = self.model.loss(x_batch, t_batch)
        history.train_loss_list.append(loss)
```

在参数更新**之后**再算一次损失，记录的是更新后的 loss。

**为什么是更新后？** 这样曲线更平滑——更新前的 loss 包含了这次梯度的信息，更新后表示"用这个 batch 学习后，在同一个 batch 上的 loss"。

（注：也有实现记录更新前的 loss，两种方式都合理，效果差别不大）

### ⑤ 打印日志

```python
        if iteration % self.print_loss_every == 0:
            epoch_num = iteration // self.iters_per_epoch + 1
            print(f"  [iter {iteration:>5d} | epoch {epoch_num}]  loss: {loss:.6f}")
```

`iteration // iters_per_epoch`：整除得到当前是第几个 Epoch（从 0 开始），+1 变成从 1 开始。

### ⑥ Epoch 结束时评估

```python
        if iteration % self.iters_per_epoch == 0:
            self._evaluate_epoch(iteration, history)
```

每当 `iteration` 是 `iters_per_epoch (294)` 的整数倍时，评估准确率。

---

## 13.5 `_evaluate_epoch`：准确率评估

```python
def _evaluate_epoch(self, iteration, history):
    train_acc = self.model.accuracy(self.x_train, self.y_train)
    # 用完整训练集评估：x_train (29400, 784)
    # 一次前向传播，得到 (29400, 10) 的预测
    
    test_acc = self.model.accuracy(self.x_test, self.y_test)
    # 用完整测试集评估：x_test (12600, 784)
    
    history.train_acc_list.append(train_acc)
    history.test_acc_list.append(test_acc)
```

**注意：评估准确率时用的是完整数据集（不是 mini-batch）**，这样得到的准确率才有参考价值。

但这也意味着每次评估时，模型要处理 29,400 + 12,600 = 42,000 张图片——一次大的前向传播。这就是为什么只在每个 Epoch 结束时评估，而不是每次迭代都评估。

---

## 13.6 训练循环的完整流程图

```
开始
  ↓
iteration = 0, 1, 2, ..., 2939
  ↓
  ① 随机取 100 个样本 → x_batch(100,784), t_batch(100,)
  ↓
  ② 前向传播 → y(100,10)；反向传播 → grads{'W1':..., ...}
  ↓
  ③ 优化器更新参数：params['W1'] -= lr * grads['W1']，...
  ↓
  ④ 计算本 batch 的 loss，追加到 train_loss_list
  ↓
  ⑤ 若 iteration % 294 == 0（Epoch 开始）：
       用全量数据评估 train_acc, test_acc，追加到列表
  ↓
iteration = 2939 后结束训练
  ↓
最终评估：打印最终 train_acc 和 test_acc
  ↓
返回 history 对象
```

---

## 13.7 训练完成后的结果

`history` 对象中：
- `train_loss_list`：长度 2940，每次迭代的 loss 曲线（理想情况是单调下降）
- `train_acc_list`：长度 11（初始 + 10 个 Epoch），训练集准确率
- `test_acc_list`：长度 11，测试集准确率

这些数据传给 `visualizer.py` 绘制训练曲线。

---

## 13.8 小结

| 概念 | 含义 |
|------|------|
| Epoch | 模型看完全部训练数据一遍 |
| Iteration | 一次 mini-batch 的前向+反向+更新 |
| Mini-batch | 每次随机取 batch_size 个样本 |
| `np.random.choice` | 有放回随机采样 |
| 参数原地更新 | 优化器直接修改 `model.params` 中的数组 |

---

[← 第 12 章](chapter-12-optimizer-code.md) | [返回目录](README.md) | [第 14 章：完整流程 →](chapter-14-main.md)
