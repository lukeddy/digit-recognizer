# 手写数字识别 — 从零实现两层神经网络

> 纯 NumPy 实现，不依赖任何深度学习框架。
> 完整覆盖：数据处理 → 前向传播 → 损失计算 → 反向传播 → 参数更新 → 可视化。

---

## 目录

1. [项目结构](#1-项目结构)
2. [快速运行](#2-快速运行)
3. [网络结构](#3-网络结构)
4. [核心理论](#4-核心理论)
   - 4.1 激活函数
   - 4.2 损失函数
   - 4.3 前向传播
   - 4.4 反向传播（链式法则）
   - 4.5 优化器
5. [代码模块详解](#5-代码模块详解)
6. [实验配置与调参](#6-实验配置与调参)
7. [理解验证方法](#7-理解验证方法)

---

## 1. 项目结构

```
digit_recognizer/
│
├── main.py          # 程序入口，串联所有模块，运行此文件即可训练
├── config.py        # 所有超参数与路径配置（唯一需要修改的文件）
│
├── dataset.py       # 数据加载与预处理（读取 CSV、归一化、划分）
├── functions.py     # 激活函数、损失函数及其导数（网络的数学基础）
├── gradient.py      # 数值梯度（中心差分法，用于验证反向传播正确性）
├── model.py         # 两层神经网络（前向传播 + 反向传播）
├── optimizer.py     # 优化器：SGD / Momentum / Adam
├── trainer.py       # 训练器（Mini-batch 训练循环）
├── visualizer.py    # 训练结果可视化（loss 曲线 + 准确率曲线）
│
└── data/
    └── train.csv    # MNIST 手写数字数据集（Kaggle 格式）
```

**模块依赖关系（调用方向）：**

```
main.py
  ├── dataset.py      ← config.py
  ├── model.py        ← functions.py, config.py
  ├── optimizer.py
  ├── trainer.py      ← model.py, optimizer.py, config.py
  └── visualizer.py   ← trainer.py (TrainHistory)
```

---

## 2. 快速运行

```bash
cd digit_recognizer
python main.py
```

训练完成后自动弹出训练曲线图，并保存到 `outputs/training_results.png`。

**切换优化器或梯度方式只需修改 `config.py`：**

```python
TRAIN_CONFIG = {
    'grad_method': 'backprop',   # 'backprop'（快）或 'numerical'（慢，用于验证）
    'optimizer':   'sgd',        # 'sgd' / 'momentum' / 'adam'
    'learning_rate': 0.1,        # SGD→0.1  Momentum→0.01  Adam→0.001
}
```

---

## 3. 网络结构

```
输入层          隐藏层             输出层
(784 节点)  →  (50 节点)      →  (10 节点)
  X         a1=X@W1+b1        a2=z1@W2+b2
            z1=sigmoid(a1)     y=softmax(a2)
```

| 层 | 节点数 | 激活函数 | 参数 |
|---|---|---|---|
| 输入层 | 784 | — | — |
| 隐藏层 | 50 | Sigmoid | W1(784×50), b1(50,) |
| 输出层 | 10 | Softmax | W2(50×10),  b2(10,) |

**为什么输入是 784？**
MNIST 每张图片是 28×28 像素的灰度图，展平后得到 784 维向量。

**为什么输出是 10？**
对应数字 0-9，共 10 个类别。Softmax 将输出转为概率分布，概率最大的类别即预测结果。

---

## 4. 核心理论

### 4.1 激活函数

激活函数的作用：在线性变换（矩阵乘法）之后引入**非线性**，否则多层叠加等价于单层线性变换，无法拟合复杂规律。

#### Sigmoid

```
σ(x) = 1 / (1 + e^{-x})
```

- 输出范围：(0, 1)
- 导数：`σ'(x) = σ(x) · (1 − σ(x))`（反向传播时用到）
- 缺点：深层网络中饱和区（输出接近 0 或 1 时）梯度趋近于 0，造成**梯度消失**

```python
# functions.py
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(z):          # z 是 sigmoid 的输出值
    return z * (1.0 - z)
```

#### ReLU

```
ReLU(x) = max(0, x)
```

- 正区间梯度恒为 1，缓解梯度消失
- 导数：x > 0 时为 1，x ≤ 0 时为 0

```python
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):             # x 是 ReLU 的输入值（激活前）
    return (x > 0).astype(float)
```

#### Softmax（输出层专用）

```
softmax(x_i) = e^{x_i} / Σ_j e^{x_j}
```

- 将任意实数向量转换为概率分布（所有输出之和为 1）
- 溢出防护：计算前减去最大值，数学上等价但避免 `e^{大数}` 溢出

```python
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)   # 溢出防护
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```

---

### 4.2 损失函数

损失函数衡量预测结果与真实标签的"差距"，训练目标是最小化损失。

#### 交叉熵损失（Cross Entropy Error）

```
L = -1/n · Σ_i log(y[i, t[i]])
```

- `y[i, t[i]]`：第 i 个样本，预测为正确类别 t[i] 的概率
- 正确类别概率越高，`-log(p)` 越小，损失越小
- 与 Softmax 配合使用，二者组合的梯度推导极为简洁

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t, y = t.reshape(1,-1), y.reshape(1,-1)
    if t.size == y.size:          # one-hot → 整数标签
        t = np.argmax(t, axis=1)
    n = y.shape[0]
    return -np.sum(np.log(y[np.arange(n), t] + 1e-7)) / n
```

---

### 4.3 前向传播

数据从输入层依次流向输出层，每层做"线性变换 + 激活"：

```
a1 = X  @ W1 + b1       # 第一层线性变换    shape: (n, 50)
z1 = sigmoid(a1)         # 第一层激活        shape: (n, 50)
a2 = z1 @ W2 + b2       # 第二层线性变换    shape: (n, 10)
y  = softmax(a2)         # 第二层激活（输出概率）shape: (n, 10)
L  = cross_entropy(y, t) # 计算损失          shape: scalar
```

对应代码：`model.py` → `TwoLayerNet.forward()`

---

### 4.4 反向传播（链式法则）

反向传播是训练神经网络的核心算法：从损失函数出发，利用**链式法则**逐层计算每个参数对损失的偏导数（梯度），然后用梯度更新参数。

#### 完整推导

**第二层（输出层）：**

Softmax + Cross-Entropy 的联合梯度有一个极简洁的结果：

```
∂L/∂a2[i, j] = y[i,j] - 1_{j == t[i]}
```

除以 n 后：

```
dy = (y - t_onehot) / n        # shape: (n, 10)
```

直觉理解：预测正确类别的概率越高，梯度越接近 0，参数调整幅度越小。

由此得到第二层参数的梯度：

```
∂L/∂W2 = z1.T @ dy            # (50, n) @ (n, 10) → (50, 10)
∂L/∂b2 = sum(dy, axis=0)      # → (10,)
∂L/∂z1 = dy  @ W2.T           # (n, 10) @ (10, 50) → (n, 50)  ← 继续往前传
```

**第一层（隐藏层）：**

梯度流过 Sigmoid 时，需要乘以 Sigmoid 的导数：

```
∂L/∂a1 = ∂L/∂z1 × sigmoid'(z1)   # 逐元素乘  shape: (n, 50)
        = ∂L/∂z1 × z1 × (1 − z1)
```

由此得到第一层参数的梯度：

```
∂L/∂W1 = X.T  @ ∂L/∂a1       # (784, n) @ (n, 50) → (784, 50)
∂L/∂b1 = sum(∂L/∂a1, axis=0) # → (50,)
```

对应代码：`model.py` → `TwoLayerNet._backprop_gradient()`

```python
def _backprop_gradient(self, X, t):
    n = X.shape[0]
    W1, b1, W2, b2 = ...

    # ① 前向传播（缓存中间值）
    a1 = X @ W1 + b1
    z1 = sigmoid(a1)
    a2 = z1 @ W2 + b2
    y  = softmax(a2)

    # ② 输出层梯度（Softmax + Cross-Entropy 联合）
    dy = y.copy()
    dy[np.arange(n), t] -= 1
    dy /= n

    # ③ 第二层参数梯度
    dW2 = z1.T @ dy
    db2 = np.sum(dy, axis=0)
    dz1 = dy @ W2.T

    # ④ 隐藏层反向（Sigmoid 导数）
    da1 = dz1 * sigmoid_grad(z1)

    # ⑤ 第一层参数梯度
    dW1 = X.T @ da1
    db1 = np.sum(da1, axis=0)

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
```

#### 数值梯度（验证工具）

用中心差分法近似导数，可验证反向传播实现是否正确：

```
f'(x) ≈ [f(x+h) - f(x-h)] / (2h)    h = 1e-4
```

两者之间的误差应小于 `1e-4`（实测误差约 `1.68e-9`，精度优秀）。

对应代码：`gradient.py` → `numerical_gradient()`

---

### 4.5 优化器

梯度告诉我们"该往哪个方向走"，优化器决定"具体怎么走"。

#### SGD（随机梯度下降）

```
θ ← θ - lr × ∇θ L
```

- 最简单的更新规则
- 对学习率敏感，损失曲面不均匀时收敛慢

```python
# optimizer.py
class SGD:
    def step(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]
```

#### Momentum（动量法）

```
v ← momentum × v  -  lr × ∇θ L
θ ← θ + v
```

- 引入速度向量 v，模拟物体运动的惯性
- 梯度方向一致时加速；方向振荡时相互抵消，减少震荡
- 典型参数：`momentum=0.9`，保留 90% 的历史速度

```python
class Momentum:
    def step(self, params, grads):
        if not self._velocity:
            for key in params:
                self._velocity[key] = np.zeros_like(params[key])
        for key in params:
            self._velocity[key] = self.momentum * self._velocity[key] - self.lr * grads[key]
            params[key] += self._velocity[key]
```

#### Adam（自适应矩估计）

```
m ← β₁·m + (1−β₁)·∇θ         # 一阶矩（梯度均值）
v ← β₂·v + (1−β₂)·(∇θ)²      # 二阶矩（梯度平方均值）

m̂ = m / (1 - β₁^t)            # 偏差修正
v̂ = v / (1 - β₂^t)

θ ← θ - lr · m̂ / (√v̂ + ε)
```

- 为每个参数自适应调整学习率：梯度大的参数学习率小，梯度小的参数学习率大
- 实践中收敛最稳定，是目前最常用的优化器
- 典型参数：`β₁=0.9, β₂=0.999, ε=1e-8, lr=0.001`

---

## 5. 代码模块详解

### `config.py` — 配置中心

所有超参数集中管理，实验时只改这一个文件：

```python
MODEL_CONFIG = {
    'input_size':  784,    # 输入维度
    'hidden_size': 50,     # 隐藏层大小（越大表达能力越强，计算也越慢）
    'output_size': 10,     # 输出类别数
    'weight_init_std': 0.01,
}

TRAIN_CONFIG = {
    'learning_rate': 0.1,
    'batch_size':    100,
    'num_epochs':    10,
    'grad_method':   'backprop',   # 'backprop' 或 'numerical'
    'optimizer':     'sgd',        # 'sgd' / 'momentum' / 'adam'
}
```

---

### `dataset.py` — 数据流水线

```
原始 CSV
  → 分离特征(X) 和标签(y)
  → train_test_split（70% 训练 / 30% 测试）
  → MinMaxScaler 归一化到 [0, 1]（在训练集上 fit，避免数据泄露）
  → 返回 (x_train, x_test, y_train, y_test)
```

**为什么归一化？**
原始像素值范围是 [0, 255]。不归一化时，大数值会导致激活函数饱和（sigmoid 输入过大时输出趋近 1，梯度接近 0），训练困难。

---

### `trainer.py` — 训练循环

```
for iteration in range(total_iters):
    ① 随机采样 Mini-batch（batch_size=100 个样本）
    ② model.gradient()  →  计算当前 batch 的梯度
    ③ optimizer.step()  →  更新所有参数
    ④ 记录当前 loss
    ⑤ 每个 Epoch 结束 → 评估训练集/测试集准确率
```

**为什么用 Mini-batch 而不是全量数据？**

| 方式 | 特点 |
|---|---|
| 全量梯度下降 | 梯度准确，但每次迭代慢；数据量大时内存不够 |
| 单样本（在线）| 更新频繁，但梯度噪声大，收敛不稳定 |
| **Mini-batch** | 折中方案，batch_size=32~256，兼顾速度与稳定性 |

---

### `model.py` — 神经网络核心

```python
class TwoLayerNet:
    params = {
        'W1': shape(784, 50),   # 第一层权重
        'b1': shape(50,),       # 第一层偏置
        'W2': shape(50, 10),    # 第二层权重
        'b2': shape(10,),       # 第二层偏置
    }

    forward(X)                  # 前向传播 → 预测概率
    loss(X, t)                  # 前向传播 → 计算损失
    accuracy(X, t)              # 前向传播 → 计算准确率
    gradient(X, t, method)      # 统一接口，内部调用 backprop 或 numerical
    _backprop_gradient(X, t)    # 反向传播（解析梯度，推荐）
    _numerical_gradient(X, t)   # 数值梯度（验证用）
```

---

## 6. 实验配置与调参

### 切换优化器

修改 `config.py` 中的 `TRAIN_CONFIG`：

| 优化器 | 推荐学习率 | 特点 |
|---|---|---|
| `sgd` | `0.1` | 简单，baseline |
| `momentum` | `0.01` | 收敛更快，推荐日常使用 |
| `adam` | `0.001` | 最稳定，对学习率不敏感 |

### 切换梯度方式

```python
'grad_method': 'backprop'    # 推荐，速度快（~50x）
'grad_method': 'numerical'   # 用于验证反向传播正确性，速度很慢
```

### 调整网络大小

```python
'hidden_size': 50    # 当前默认，适合入门
'hidden_size': 100   # 表达能力更强，可能提升准确率
'hidden_size': 10    # 太小，容易欠拟合
```

---

## 7. 理解验证方法

### 如何确认反向传播实现正确？

用数值梯度与解析梯度对比，误差应 < 1e-4：

```python
from model import TwoLayerNet
from gradient import numerical_gradient
import numpy as np

net = TwoLayerNet()
X = np.random.randn(8, 784)
t = np.random.randint(0, 10, 8)

bp_grads  = net._backprop_gradient(X, t)
num_grads = {k: numerical_gradient(lambda _: net.loss(X, t), net.params[k])
             for k in ['W1', 'b1']}  # 只验证部分参数（数值梯度慢）

for key in num_grads:
    diff = np.max(np.abs(bp_grads[key] - num_grads[key]))
    print(f"{key}: 最大误差 = {diff:.2e}")   # 预期 < 1e-4
```

### 关键概念对照表

| 概念 | 代码位置 | 关键行 |
|---|---|---|
| 前向传播 | `model.py:forward()` | `a1 = X @ W1 + b1` |
| Softmax+CE 联合梯度 | `model.py:_backprop_gradient()` | `dy[np.arange(n), t] -= 1` |
| Sigmoid 反向 | `model.py:_backprop_gradient()` | `da1 = dz1 * sigmoid_grad(z1)` |
| 参数更新 | `optimizer.py:SGD.step()` | `params[key] -= lr * grads[key]` |
| Mini-batch 采样 | `trainer.py:train()` | `np.random.choice(n, batch_size)` |
| 数值梯度验证 | `gradient.py` | `(f(x+h) - f(x-h)) / (2*h)` |


---

还缺什么（进阶基础）                                      
                                                                                                                                                                      
  如果要说"完整"，还差几个常见概念：
                                                                                                                                                                      
  1. 权重初始化策略                                         
  目前只用了 std * random_normal，实际教学中还有：                                                                                                                    
  - Xavier 初始化（配合 sigmoid）：std = sqrt(1 / n_in)                                                                                                               
  - He 初始化（配合 ReLU）：std = sqrt(2 / n_in)                                                                                                                      
                                                                                                                                                                      
  2. 正则化                                                                                                                                                           
  - L2 正则化（Weight Decay）：防止过拟合，损失函数中加 λ/2 * ||W||²
  - Dropout：随机丢弃神经元，工程中更常用                                                                                                                             
   
  3. 批归一化（Batch Normalization）                                                                                                                                  
  目前的网络训练深层时容易不稳定，BatchNorm 是现代网络的标配基础组件
                                                                                                                                                                      
  4. 多层扩展                                               
  目前是固定的两层，封装成可变层数（比如 [784, 100, 50, 10]）会更通用 
