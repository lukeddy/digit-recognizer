# 第 10 章：激活与损失函数实现（functions.py）

> 这章把第 3 章的理论翻译成 numpy 代码，重点关注数值稳定性和实现细节。

## 10.1 文件定位

`functions.py` 是一个**纯函数库**——它只包含无状态的函数，不含任何类或全局变量。所有函数都接收 numpy 数组，返回 numpy 数组。

---

## 10.2 sigmoid 函数

```python
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
```

**维度行为：**

```python
# 单个值
sigmoid(0.0)        # → 0.5（标量）

# 向量（逐元素）
sigmoid(np.array([0.0, 1.0, -1.0]))
# → array([0.5, 0.731, 0.269])   shape: (3,)

# 矩阵（逐元素）
sigmoid(np.array([[0.1, -0.2], [0.3, 0.4]]))
# → shape: (2, 2)，每个元素独立计算
```

关键：**输入是什么 shape，输出就是什么 shape**（逐元素运算）。

---

## 10.3 sigmoid_grad 函数

```python
def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    return z * (1.0 - z)
```

**注意参数名是 `z`，不是 `x`**——这里传入的是 sigmoid 的**输出值**（已经算好的激活值），而不是输入。

对比两种写法：

```python
# 方式1：传入 sigmoid 输入值 x，每次都要重新计算 sigmoid
def sigmoid_grad_v1(x):
    s = sigmoid(x)
    return s * (1 - s)   # 多了一次 sigmoid 计算

# 方式2（实际使用）：传入 sigmoid 输出值 z，直接用
def sigmoid_grad(z):
    return z * (1.0 - z)  # z 已经算好了，直接用
```

在反向传播中，我们已经在前向传播时计算并缓存了 `z1 = sigmoid(a1)`，所以直接用 `z1` 调用 `sigmoid_grad(z1)` 可以节省一次计算。

---

## 10.4 relu 和 relu_grad 函数

```python
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)     # 取 0 和 x 中较大的

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)  # x>0 的位置是1.0，否则0.0
```

**`relu_grad` 传入的是输入值 `x`（不是输出）**——因为需要知道哪些位置是正数（梯度=1）。

对比 sigmoid_grad：sigmoid_grad 传**输出**（z），relu_grad 传**输入**（x）。原因：
- sigmoid 的导数 $\sigma(x)(1-\sigma(x))$ 可以用输出 $z$ 直接表达
- ReLU 的导数 $\mathbf{1}[x>0]$ 需要判断**原始输入**的正负

---

## 10.5 softmax 函数（含批量处理）

```python
def softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.T                              # (n, 10) → (10, n)
        x = x - np.max(x, axis=0)           # 每列减最大值（防止溢出）
        y = np.exp(x) / np.sum(np.exp(x), axis=0)  # 每列归一化
        return y.T                           # (10, n) → (n, 10)
    # 单样本情况
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```

### 为什么要转置？

处理 2D 输入（n 个样本）时，需要**对每一行**（每个样本的 10 个类别）单独做 softmax。

numpy 的广播规则对"对每列操作"更自然（axis=0），所以先转置让每个样本变成一列，操作完再转回去。

让我们逐步看：

```python
x = a2   # shape: (n, 10)，n 个样本，10 个类别

# 转置
x = x.T  # shape: (10, n)，每列是一个样本的 10 个分数

# 每列减去该列最大值（防止溢出）
np.max(x, axis=0)   # shape: (n,)，找出每列（每个样本）的最大值
x = x - np.max(x, axis=0)   # (10, n) - (n,) → 广播，(10, n)
                              # 每列的每个元素都减去该列的最大值

# 每列归一化
np.exp(x)                   # shape: (10, n)，每元素取指数
np.sum(np.exp(x), axis=0)   # shape: (n,)，每列的指数和
y = np.exp(x) / np.sum(np.exp(x), axis=0)   # (10, n) / (n,) → (10, n)

# 转回来
y.T   # shape: (n, 10)，结果
```

**等价的、更直观的写法（但性能稍差）：**

```python
# 对每行做 softmax（语义更清晰，但 axis 的写法不同）
x = a2 - np.max(a2, axis=1, keepdims=True)   # (n, 10) - (n, 1) → (n, 10)
y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)  # (n, 10)
```

---

## 10.6 cross_entropy_error 函数

```python
def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
    if t.size == y.size:     # t 是 one-hot 编码 → 转为整数标签
        t = np.argmax(t, axis=1)
    n = y.shape[0]
    return -np.sum(np.log(y[np.arange(n), t] + 1e-7)) / n
```

### 关键代码详解

#### `y.ndim == 1` 的处理

如果传入的是单个样本（1D），变成 2D：

```python
y = np.array([0.1, 0.2, 0.7])    # shape: (3,)
t = np.array([2])                  # shape: (1,)

# 处理后
y → shape: (1, 3)
t → shape: (1, 1)
```

#### `t.size == y.size` 的处理

t 有两种格式：
- **整数标签**：`t = [2, 0, 1]`，shape = `(n,)`
- **one-hot 编码**：`t = [[0,0,1], [1,0,0], [0,1,0]]`，shape = `(n, K)`

如果是 one-hot（`t.size == y.size`），用 `argmax` 转成整数标签：

```python
t = np.array([[0, 0, 1],    # 第0个样本是类别2
              [1, 0, 0],    # 第1个样本是类别0
              [0, 1, 0]])   # 第2个样本是类别1

np.argmax(t, axis=1)   # → [2, 0, 1]
```

#### 花式索引 `y[np.arange(n), t]`

```python
n = 3
np.arange(n)   # → [0, 1, 2]
t              # → [2, 0, 1]

y[np.arange(n), t]
# = y[[0, 1, 2], [2, 0, 1]]
# = [y[0,2], y[1,0], y[2,1]]
# = [0.7, 0.8, 0.6]   ← 各样本正确类别的预测概率
```

这一行等价于"取出 y 矩阵中，第 i 行、第 t[i] 列的元素"——正是我们需要的正确类别概率。

#### `+ 1e-7` 的意义

如果某个概率 `y[i, t[i]] = 0`（模型完全没有预测到正确类别），`log(0) = -∞` 会导致损失变成无穷大，引发数值错误。加上 $10^{-7}$ 避免这个情况：

```python
np.log(0 + 1e-7) = np.log(1e-7) ≈ -16.1   # 很大但不是无穷
```

---

## 10.7 函数在项目中的调用关系

```python
# model.py 的 forward 方法中：
z1 = sigmoid(a1)    # functions.py 的 sigmoid
y  = softmax(a2)    # functions.py 的 softmax

# model.py 的 loss 方法中：
y = self.forward(X)
L = cross_entropy_error(y, t)   # functions.py 的 cross_entropy_error

# model.py 的反向传播中：
da1 = dz1 * sigmoid_grad(z1)   # functions.py 的 sigmoid_grad
```

---

## 10.8 小结

| 函数 | 输入 shape | 输出 shape | 注意点 |
|------|-----------|-----------|--------|
| `sigmoid(x)` | 任意 | 同输入 | 逐元素，值在 (0,1) |
| `sigmoid_grad(z)` | 任意（sigmoid 的输出）| 同输入 | 传**输出值** |
| `relu(x)` | 任意 | 同输入 | 逐元素，max(0,x) |
| `relu_grad(x)` | 任意（relu 的输入）| 同输入 | 传**输入值** |
| `softmax(x)` | `(n, K)` 或 `(K,)` | 同输入 | 行内归一化，含溢出防护 |
| `cross_entropy_error(y, t)` | `(n, K)`, `(n,)` | 标量 | 含 +1e-7 防护 |

---

[← 第 9 章](chapter-09-dataset.md) | [返回目录](README.md) | [第 11 章：模型实现 →](chapter-11-model.md)
