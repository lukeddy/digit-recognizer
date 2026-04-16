# 第 11 章：模型实现（model.py）

> 这是整个项目的核心文件。把第 4 章（前向传播）和第 6 章（反向传播）的理论转化为代码。

## 11.1 TwoLayerNet 类总览

```python
class TwoLayerNet:
    def __init__(self, ...)      # 初始化：创建权重和偏置
    def forward(self, X)         # 前向传播：X → 概率
    def loss(self, X, t)         # 损失：调用 forward，再算交叉熵
    def accuracy(self, X, t)     # 准确率：取最大概率的类别
    def gradient(self, X, t, method)    # 梯度：数值/反向传播
    def _numerical_gradient(self, X, t) # 数值梯度（调试用）
    def _backprop_gradient(self, X, t)  # 反向传播（正式训练用）
```

---

## 11.2 `__init__`：参数初始化

```python
def __init__(
    self,
    input_size=784,
    hidden_size=50,
    output_size=10,
    weight_init_std=0.01,
):
    self.params = {
        'W1': np.random.randn(input_size, hidden_size) * weight_init_std,
        'b1': np.zeros(hidden_size),
        'W2': np.random.randn(hidden_size, output_size) * weight_init_std,
        'b2': np.zeros(output_size),
    }
```

### 参数形状

```
W1: (784, 50)   ← 输入层(784) → 隐藏层(50) 的连接权重
b1: (50,)       ← 隐藏层 50 个节点各自的偏置
W2: (50, 10)    ← 隐藏层(50) → 输出层(10) 的连接权重
b2: (10,)       ← 输出层 10 个节点各自的偏置
```

权重矩阵的 shape 规律：

```
层 l 的权重 W：shape = (上一层节点数, 本层节点数)
                         ↑               ↑
                      输入维度          输出维度
```

这样矩阵乘法 `X @ W` 才能成立：
```
X(n, 上一层) @ W(上一层, 本层) = 结果(n, 本层)
```

### 为什么偏置初始化为 0？

偏置初始化为 0 是标准做法——它只控制激活函数的"触发阈值"，没有打破对称性的需求。

### 为什么权重用随机小值？

**随机**：避免所有权重相同，否则所有神经元学到的东西完全一样（对称性问题）。每个神经元需要从不同的起始点出发，学到不同的特征。

**乘以 0.01（小值）**：防止激活函数饱和。Sigmoid 在输入 `|x| >> 0` 时梯度趋近于 0（梯度消失）。初始值小，`W @ X` 的结果在 0 附近，Sigmoid 工作在梯度最大的区域。

---

## 11.3 `forward`：前向传播

```python
def forward(self, X: np.ndarray) -> np.ndarray:
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    a1 = X @ W1 + b1     # shape: (n, 784)@(784, 50)+(50,) = (n, 50)
    z1 = sigmoid(a1)     # shape: (n, 50)，逐元素

    a2 = z1 @ W2 + b2    # shape: (n, 50)@(50, 10)+(10,) = (n, 10)
    y  = softmax(a2)     # shape: (n, 10)，逐行归一化

    return y
```

代码与数学公式一一对应：

| 数学 | 代码 | Shape |
|------|------|-------|
| $a_1 = X W_1 + b_1$ | `a1 = X @ W1 + b1` | `(n, 50)` |
| $z_1 = \sigma(a_1)$ | `z1 = sigmoid(a1)` | `(n, 50)` |
| $a_2 = z_1 W_2 + b_2$ | `a2 = z1 @ W2 + b2` | `(n, 10)` |
| $y = \text{softmax}(a_2)$ | `y = softmax(a2)` | `(n, 10)` |

---

## 11.4 `loss` 和 `accuracy`

```python
def loss(self, X, t):
    y = self.forward(X)              # (n, 10)
    return cross_entropy_error(y, t) # 标量

def accuracy(self, X, t):
    y_proba = self.forward(X)        # (n, 10)
    y_pred  = np.argmax(y_proba, axis=1)  # (n,) ← 取每行最大值的索引
    return float(np.mean(y_pred == t))    # 标量
```

### `np.argmax(y_proba, axis=1)` 详解

```python
y_proba = [[0.02, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.86, 0.01, 0.01],  # 样本0
            [0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00],  # 样本1
            [0.01, 0.02, 0.01, 0.01, 0.92, 0.01, 0.01, 0.01, 0.00, 0.00]]  # 样本2
            
np.argmax(y_proba, axis=1)   # axis=1：对每一行找最大值的位置
# → [7, 0, 4]
# 样本0预测为7，样本1预测为0，样本2预测为4
```

`axis=1` 意为"沿第 1 维（列方向）找最大值"，即对每行找最大值的列索引。

### `np.mean(y_pred == t)` 详解

```python
y_pred = [7, 0, 4]
t      = [7, 0, 2]   # 真实标签

y_pred == t          # → [True, True, False]
np.mean([True, True, False])  # True=1, False=0
# → 2/3 ≈ 0.667
```

`np.mean` 计算 True 的比例 = 预测正确的样本比例 = 准确率。

---

## 11.5 `_backprop_gradient`：反向传播（核心）

```python
def _backprop_gradient(self, X, t):
    n = X.shape[0]
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
```

### ① 前向传播（缓存中间值）

```python
    a1 = X @ W1 + b1    # (n, 50)
    z1 = sigmoid(a1)    # (n, 50)  ← 缓存！sigmoid_grad 需要用
    a2 = z1 @ W2 + b2   # (n, 10)
    y  = softmax(a2)    # (n, 10)
```

反向传播需要前向传播时的中间值：
- `z1`：sigmoid_grad 需要它（`sigmoid_grad(z1) = z1*(1-z1)`）
- `y`：计算输出层梯度需要它

### ② 输出层梯度（softmax + 交叉熵联合）

```python
    dy = y.copy()              # (n, 10)，先复制 y
    dy[np.arange(n), t] -= 1  # 在正确类别的位置减 1
    dy /= n                    # 除以 n 取平均
```

回顾第 6 章推导的结论：

$$\frac{\partial L}{\partial a_2[i, j]} = \frac{1}{n}(y[i,j] - \mathbf{1}[j = t[i]])$$

代码翻译：
- `dy = y.copy()`：初始化，所有位置都是 $y[i,j]$
- `dy[np.arange(n), t] -= 1`：在正确类别位置（$j = t[i]$）减去 1（实现 $-\mathbf{1}[j=t[i]]$）
- `dy /= n`：除以 n 取平均

**数值示例：**

```
n = 3，t = [7, 0, 4]

y（前向传播输出）:
  [[0.02, 0.01, ..., 0.86, ...],    # 样本0，第7列是正确类别
   [0.95, 0.01, ..., 0.00, ...],    # 样本1，第0列是正确类别
   [0.01, 0.02, ..., 0.92, ...]]    # 样本2，第4列是正确类别

dy = y.copy() 后：
  同上

dy[np.arange(3), t] -= 1：
  dy[[0,1,2], [7,0,4]] -= 1
  → dy[0,7] = 0.86 - 1 = -0.14     # 正确类别（概率离1还差0.14）
     dy[1,0] = 0.95 - 1 = -0.05    # 正确类别
     dy[2,4] = 0.92 - 1 = -0.08    # 正确类别
  其他位置不变

dy /= 3：所有值除以3
```

### ③ 第二层参数梯度

```python
    dW2 = z1.T @ dy            # (50, n) @ (n, 10) = (50, 10)
    db2 = np.sum(dy, axis=0)   # 对 n 行求和 → (10,)
```

**维度验证：**
```
dW2:
z1.T: (50, 29400)  [在评估时 n 很大，这里用 29400 示意]
dy:   (29400, 10)
结果: (50, 10)   ← 和 W2 shape 一致 ✓

db2:
np.sum(dy, axis=0): (10,)   ← 和 b2 shape 一致 ✓
```

### ④ 梯度向上传播

```python
    dz1 = dy @ W2.T            # (n, 10) @ (10, 50) = (n, 50)
```

**维度验证：**
```
dy:   (n, 10)
W2.T: (10, 50)   ← W2 是 (50, 10)，转置后是 (10, 50)
结果: (n, 50)   ← 和 z1 shape 一致 ✓
```

### ⑤ sigmoid 反向传播

```python
    da1 = dz1 * sigmoid_grad(z1)  # 逐元素乘法
```

**维度验证：**
```
dz1:              (n, 50)
sigmoid_grad(z1): (n, 50)   ← z1 是 (n,50)，导数和它同 shape
da1:              (n, 50)   ← 逐元素乘，shape 不变 ✓
```

### ⑥ 第一层参数梯度

```python
    dW1 = X.T @ da1            # (784, n) @ (n, 50) = (784, 50)
    db1 = np.sum(da1, axis=0)  # 对 n 行求和 → (50,)
```

**维度验证：**
```
dW1:
X.T:  (784, n)
da1:  (n, 50)
结果: (784, 50)   ← 和 W1 shape 一致 ✓

db1:
np.sum(da1, axis=0): (50,)   ← 和 b1 shape 一致 ✓
```

### 返回梯度字典

```python
    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
```

返回的字典结构和 `self.params` 完全一致，优化器可以直接遍历使用。

---

## 11.6 完整维度流图

```
前向传播:                                反向传播:
                                          ↓
X(n,784)                          db1=(50,) ← sum(da1,axis=0)
  ↓@W1(784,50)                    dW1=(784,50)←X.T(784,n)@da1(n,50)
a1(n,50)                                  ↑
  ↓sigmoid                        da1(n,50)←dz1(n,50)*sigmoid_grad(z1)(n,50)
z1(n,50)                                  ↑
  ↓@W2(50,10)                     dz1(n,50)←dy(n,10)@W2.T(10,50)
a2(n,10)                                  ↑
  ↓softmax                  db2=(10,)← sum(dy,axis=0)
y(n,10)                     dW2=(50,10)←z1.T(50,n)@dy(n,10)
  ↓cross_entropy              ↑
L(标量)          dy(n,10) = (y-T)/n ← 从这里开始反向传播
```

---

## 11.7 小结

| 方法 | 输入 | 输出 | 关键维度 |
|------|------|------|----------|
| `__init__` | 配置 | - | W1:(784,50), W2:(50,10) |
| `forward(X)` | `(n,784)` | `(n,10)` | 两层线性+激活 |
| `loss(X,t)` | `(n,784)`, `(n,)` | 标量 | 调用 forward |
| `accuracy(X,t)` | `(n,784)`, `(n,)` | 标量 | argmax 取预测类别 |
| `_backprop_gradient` | `(n,784)`, `(n,)` | 梯度字典 | 全部梯度同参数 shape |

---

[← 第 10 章](chapter-10-functions.md) | [返回目录](README.md) | [第 12 章：优化器实现 →](chapter-12-optimizer-code.md)
