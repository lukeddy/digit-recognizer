# 第 6 章：反向传播

> 这是整本小册子最核心的章节。反向传播是神经网络训练的引擎，理解它意味着你真正理解了深度学习的本质。

## 6.1 问题：参数如何更新？

训练的目标是让损失 $L$ 尽可能小。我们有 39,760 个参数（W1, b1, W2, b2），每次计算完损失后，需要知道：

> **每个参数应该增大还是减小，才能让 L 减小？**

答案就是**梯度**（Gradient）：$\frac{\partial L}{\partial \theta}$

梯度告诉我们，当参数 $\theta$ 增大一点点时，$L$ 会如何变化：
- 梯度 > 0：L 随 $\theta$ 增大而增大 → 应该**减小** $\theta$
- 梯度 < 0：L 随 $\theta$ 增大而减小 → 应该**增大** $\theta$
- 梯度 = 0：当前参数处于极值点

**反向传播**就是高效计算所有参数梯度的算法。

---

## 6.2 链式法则：反向传播的数学基础

反向传播的核心数学工具是**链式法则**（Chain Rule）。

### 6.2.1 单变量链式法则

如果 $z = f(y)$，$y = g(x)$，即 $z = f(g(x))$，那么：

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

### 6.2.2 直观理解

想象一个工厂流水线：$x \rightarrow y \rightarrow z$

- 当 x 变化 1 单位时，y 变化 $\frac{dy}{dx}$ 单位
- 当 y 变化 1 单位时，z 变化 $\frac{dz}{dy}$ 单位
- 所以当 x 变化 1 单位时，z 变化 $\frac{dy}{dx} \times \frac{dz}{dy}$ 单位

**反向传播就是把这个"链条"从右到左依次计算。**

### 6.2.3 在神经网络中

我们的计算图（从左到右）：

```
X → [a1 = X@W1+b1] → [z1 = σ(a1)] → [a2 = z1@W2+b2] → [y = softmax(a2)] → [L]
```

反向传播（从右到左）计算每个参数对 L 的贡献：

```
L → ∂L/∂a2 → ∂L/∂z1 → ∂L/∂a1 → ∂L/∂W1, ∂L/∂b1
              ↓
          ∂L/∂W2, ∂L/∂b2
```

---

## 6.3 符号约定

为了简洁，我们用简写：

| 简写 | 完整含义 |
|------|----------|
| $dy$ | $\frac{\partial L}{\partial a_2}$，shape `(n, 10)` |
| $dz_1$ | $\frac{\partial L}{\partial z_1}$，shape `(n, 50)` |
| $da_1$ | $\frac{\partial L}{\partial a_1}$，shape `(n, 50)` |
| $dW_2$ | $\frac{\partial L}{\partial W_2}$，shape `(50, 10)` |
| $db_2$ | $\frac{\partial L}{\partial b_2}$，shape `(10,)` |
| $dW_1$ | $\frac{\partial L}{\partial W_1}$，shape `(784, 50)` |
| $db_1$ | $\frac{\partial L}{\partial b_1}$，shape `(50,)` |

---

## 6.4 第一步：输出层梯度（Softmax + Cross-Entropy 联合推导）

这是最精彩的部分。我们要推导 $\frac{\partial L}{\partial a_2}$。

### 6.4.1 设置

- $a_2 \in \mathbb{R}^{n \times 10}$：第二层线性输出
- $y = \text{softmax}(a_2)$，$y \in \mathbb{R}^{n \times 10}$
- $L = -\frac{1}{n}\sum_{i=1}^{n} \log(y[i, t[i]])$

我们只关注**单个样本**（下标 $i$ 省略），之后除以 $n$ 即可。

单个样本的损失：$\ell = -\log(y_{t})$，其中 $y_t = y[t]$ 是正确类别的预测概率。

### 6.4.2 对 softmax 输出 y 的梯度

$$\frac{\partial \ell}{\partial y_j} = \begin{cases} -\frac{1}{y_t} & j = t \\ 0 & j \neq t \end{cases}$$

因为 $\ell = -\log(y_t)$，只和 $y_t$ 有关，对其他 $y_j$ 的偏导为 0。

### 6.4.3 softmax 对输入 a₂ 的导数

softmax 的输出 $y_j = \frac{e^{a_j}}{\sum_k e^{a_k}}$

对输入 $a_s$ 求偏导（分两种情况，因为 $\sum_k e^{a_k}$ 包含 $a_s$）：

**当 $j = s$ 时（同一个位置）：**

用商的求导法则，令 $S = \sum_k e^{a_k}$：

$$\frac{\partial y_j}{\partial a_s} = \frac{e^{a_s} \cdot S - e^{a_s} \cdot e^{a_s}}{S^2} = \frac{e^{a_s}}{S} - \frac{e^{a_s}}{S}\cdot\frac{e^{a_s}}{S} = y_j - y_j^2 = y_j(1 - y_j)$$

**当 $j \neq s$ 时（不同位置）：**

$$\frac{\partial y_j}{\partial a_s} = \frac{0 \cdot S - e^{a_j} \cdot e^{a_s}}{S^2} = -\frac{e^{a_j}}{S}\cdot\frac{e^{a_s}}{S} = -y_j \cdot y_s$$

### 6.4.4 联合推导（链式法则）

$$\frac{\partial \ell}{\partial a_s} = \sum_{j=0}^{9} \frac{\partial \ell}{\partial y_j} \cdot \frac{\partial y_j}{\partial a_s}$$

把求和拆开，$j = t$ 和 $j \neq t$ 两部分：

$$= \frac{\partial \ell}{\partial y_t} \cdot \frac{\partial y_t}{\partial a_s} + \sum_{j \neq t} \frac{\partial \ell}{\partial y_j} \cdot \frac{\partial y_j}{\partial a_s}$$

代入刚才求的偏导（注意 $j \neq t$ 时 $\frac{\partial \ell}{\partial y_j} = 0$，所以第二项消掉）：

$$= \left(-\frac{1}{y_t}\right) \cdot \frac{\partial y_t}{\partial a_s} + 0$$

**情况1：$s = t$（对正确类别的输入求导）：**

$$\frac{\partial \ell}{\partial a_t} = -\frac{1}{y_t} \cdot y_t(1 - y_t) = -(1 - y_t) = y_t - 1$$

**情况2：$s \neq t$（对其他类别的输入求导）：**

$$\frac{\partial \ell}{\partial a_s} = -\frac{1}{y_t} \cdot (-y_t \cdot y_s) = y_s$$

### 6.4.5 合并成一个公式

$$\frac{\partial \ell}{\partial a_j} = \begin{cases} y_j - 1 & j = t \\ y_j & j \neq t \end{cases} = y_j - \mathbf{1}[j = t]$$

其中 $\mathbf{1}[j = t]$ 是指示函数：当 $j = t$ 时等于 1，否则等于 0。

用向量形式（对单个样本）：

$$\frac{\partial \ell}{\partial \mathbf{a}_2} = \mathbf{y} - \mathbf{e}_t$$

其中 $\mathbf{e}_t$ 是第 $t$ 个位置为 1 的 one-hot 向量。

### 6.4.6 推广到批量（除以 n）

对 $n$ 个样本求平均损失，对应梯度也除以 $n$：

$$\boxed{dy = \frac{\partial L}{\partial a_2} = \frac{1}{n}(y - T)}$$

其中 $T$ 是 one-hot 标签矩阵，shape `(n, 10)`，$T[i, t[i]] = 1$，其余为 0。

**代码实现：**

```python
n = X.shape[0]
dy = y.copy()             # shape (n, 10)，先复制预测概率
dy[np.arange(n), t] -= 1  # 正确类别的位置减 1
dy /= n                   # 除以批量大小得平均梯度
```

**这段代码几乎是数学公式的直译！**

---

## 6.5 第二步：第二层参数的梯度

已知 $dy = \frac{\partial L}{\partial a_2}$，shape `(n, 10)`。

前向传播：$a_2 = z_1 \cdot W_2 + b_2$

### W₂ 的梯度

对于矩阵乘法 $a_2 = z_1 \cdot W_2$，利用矩阵微积分的结论（这里给出结论，附完整证明）：

$$\frac{\partial L}{\partial W_2} = z_1^T \cdot dy$$

**维度验证：**
```
z1.T @ dy:
z1.T: (50, n)
dy:   (n, 10)
结果: (50, 10)    ← 和 W2 的 shape 相同 ✓
```

**直观理解：** 如果 $z_1$ 中某个特征值很大（它的激活程度高），而 $dy$ 中对应的梯度也大，那么 $W_2$ 中连接它们的权重对损失影响大，因此梯度也应该大。转置是为了对齐维度。

**矩阵梯度的完整推导（选读）：**

考虑 $a_2$ 的第 $i$ 行第 $j$ 列：

$$a_2[i, j] = \sum_{k} z_1[i, k] \cdot W_2[k, j]$$

对 $W_2[p, q]$ 求偏导：

$$\frac{\partial a_2[i, j]}{\partial W_2[p, q]} = \begin{cases} z_1[i, p] & j = q \\ 0 & j \neq q \end{cases}$$

所以：

$$\frac{\partial L}{\partial W_2[p, q]} = \sum_{i, j} \frac{\partial L}{\partial a_2[i, j]} \cdot \frac{\partial a_2[i, j]}{\partial W_2[p, q]} = \sum_{i} dy[i, q] \cdot z_1[i, p]$$

写成矩阵形式：$\frac{\partial L}{\partial W_2} = z_1^T \cdot dy$，即 shape `(50, n) @ (n, 10) = (50, 10)` ✓

### b₂ 的梯度

前向传播是 $a_2 = \text{(某矩阵)} + b_2$（广播加法），对每个样本都加了同一个 $b_2$。

所以 $b_2$ 的梯度是把 $n$ 个样本的梯度**对行求和**：

$$\frac{\partial L}{\partial b_2} = \sum_{i=1}^{n} dy[i, :] = \text{np.sum}(dy, \text{axis=0})$$

**维度验证：**
```
np.sum(dy, axis=0):
dy: (n, 10), 对 n 行求和 → (10,)   ← 和 b2 的 shape 相同 ✓
```

### 传回隐藏层的梯度

前向传播：$a_2 = z_1 \cdot W_2$，利用对称的矩阵微积分结论：

$$\frac{\partial L}{\partial z_1} = dy \cdot W_2^T$$

**维度验证：**
```
dy @ W2.T:
dy:   (n, 10)
W2.T: (10, 50)   ← W2 的转置
结果: (n, 50)    ← 和 z1 的 shape 相同 ✓
```

**直观理解：** 每个隐藏节点 $z_1[i, k]$ 的梯度 = 它影响的所有输出节点梯度的加权和（权重就是 $W_2$）。转置是因为方向反了。

---

## 6.6 第三步：隐藏层的梯度（Sigmoid 反向传播）

已知 $dz_1 = \frac{\partial L}{\partial z_1}$，shape `(n, 50)`。

前向传播：$z_1 = \sigma(a_1)$

根据链式法则：

$$\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial z_1} \cdot \frac{\partial z_1}{\partial a_1}$$

$z_1$ 是对 $a_1$ 逐元素应用 sigmoid，所以 $\frac{\partial z_1[i,j]}{\partial a_1[i,j]} = \sigma'(a_1[i,j]) = z_1[i,j](1 - z_1[i,j])$

写成矩阵形式（逐元素乘法）：

$$da_1 = dz_1 \cdot (z_1 \cdot (1 - z_1))$$

```python
da1 = dz1 * sigmoid_grad(z1)    # 逐元素乘，shape不变
# sigmoid_grad(z1) = z1 * (1 - z1)
```

**维度验证：**
```
dz1:             (n, 50)
sigmoid_grad(z1): (n, 50)   ← z1 是 sigmoid 输出，shape 和 a1 一样
da1:             (n, 50)    ← 逐元素乘，shape不变 ✓
```

---

## 6.7 第四步：第一层参数的梯度

已知 $da_1$，前向传播是 $a_1 = X \cdot W_1 + b_1$，与第二层完全对称：

$$dW_1 = X^T \cdot da_1$$

**维度验证：**
```
X.T @ da1:
X.T:  (784, n)
da1:  (n, 50)
结果: (784, 50)   ← 和 W1 的 shape 相同 ✓
```

$$db_1 = \text{np.sum}(da_1, \text{axis=0})$$

**维度验证：**
```
np.sum(da1, axis=0):
da1: (n, 50), 对 n 行求和 → (50,)   ← 和 b1 的 shape 相同 ✓
```

---

## 6.8 完整反向传播总结

前向传播（左 → 右）：

```
X(n,784) → a1(n,50) → z1(n,50) → a2(n,10) → y(n,10) → L(标量)
```

反向传播（右 → 左）：

```
① 输出层（softmax + 交叉熵联合）：
   dy = (y - T) / n                        shape: (n, 10)

② 第二层参数：
   dW2 = z1.T @ dy                         shape: (50, 10)  ← 和 W2 相同
   db2 = sum(dy, axis=0)                   shape: (10,)     ← 和 b2 相同

③ 向上传梯度：
   dz1 = dy @ W2.T                         shape: (n, 50)   ← 和 z1 相同

④ sigmoid 反向：
   da1 = dz1 * z1*(1-z1)                   shape: (n, 50)   ← 和 a1 相同

⑤ 第一层参数：
   dW1 = X.T @ da1                         shape: (784, 50) ← 和 W1 相同
   db1 = sum(da1, axis=0)                  shape: (50,)     ← 和 b1 相同
```

**黄金规律：每个参数的梯度形状，永远和该参数本身的形状相同。**

---

## 6.9 代码实现逐行解析

```python
def _backprop_gradient(self, X, t):
    n = X.shape[0]
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    # ① 前向传播（缓存中间值，反向用）
    a1 = X @ W1 + b1          # (n, 50)
    z1 = sigmoid(a1)           # (n, 50)
    a2 = z1 @ W2 + b2          # (n, 10)
    y  = softmax(a2)           # (n, 10)

    # ② 输出层梯度（softmax + cross-entropy 联合）
    dy = y.copy()              # (n, 10)
    dy[np.arange(n), t] -= 1  # 正确类别位置 -1（构造 y - T）
    dy /= n                    # 除以 n 取平均

    # ③ 第二层参数梯度
    dW2 = z1.T @ dy            # (50,n)@(n,10) = (50,10)
    db2 = np.sum(dy, axis=0)   # (10,)

    # ④ 梯度传回隐藏层
    dz1 = dy @ W2.T            # (n,10)@(10,50) = (n,50)

    # ⑤ sigmoid 反向传播
    da1 = dz1 * sigmoid_grad(z1)  # (n,50) ⊙ (n,50) = (n,50)

    # ⑥ 第一层参数梯度
    dW1 = X.T @ da1            # (784,n)@(n,50) = (784,50)
    db1 = np.sum(da1, axis=0)  # (50,)

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
```

---

## 6.10 数值梯度 vs 反向传播

### 数值梯度（验证用）

用中心差分法近似导数：

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + h) - L(\theta_i - h)}{2h}$$

- 对**每个参数**，分别做两次前向传播
- 39,760 个参数 → 约 80,000 次前向传播
- 慢，但无需手动推导，用于**验证反向传播是否正确**

### 反向传播

- 只需**一次**前向传播 + **一次**反向传播
- 速度快约 1000 倍
- 是实际训练所用的方法

---

## 6.11 小结

| 步骤 | 公式 | Shape |
|------|------|-------|
| 输出层梯度 | `dy = (y - T) / n` | `(n, 10)` |
| dW2 | `z1.T @ dy` | `(50, 10)` |
| db2 | `sum(dy, axis=0)` | `(10,)` |
| 传回梯度 | `dz1 = dy @ W2.T` | `(n, 50)` |
| sigmoid 反传 | `da1 = dz1 * z1*(1-z1)` | `(n, 50)` |
| dW1 | `X.T @ da1` | `(784, 50)` |
| db1 | `sum(da1, axis=0)` | `(50,)` |

> **下一章**：有了梯度，怎么用它来更新参数？SGD、Momentum、Adam 三种策略。

---

[← 第 5 章](chapter-05-loss-functions.md) | [返回目录](README.md) | [第 7 章：优化器 →](chapter-07-optimizers.md)
