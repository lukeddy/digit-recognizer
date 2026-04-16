# 第 12 章：优化器实现（optimizer.py）

> 把第 7 章的三种优化器公式翻译成代码，同时学习"策略模式"这种设计思想。

## 12.1 设计思想：策略模式

`optimizer.py` 用了一种叫**策略模式**（Strategy Pattern）的设计：

```python
class BaseOptimizer(ABC):      # 抽象基类（"接口"）
    @abstractmethod
    def step(self, params, grads): ...

class SGD(BaseOptimizer):      # 具体策略1
    def step(self, params, grads): ...

class Momentum(BaseOptimizer): # 具体策略2
    def step(self, params, grads): ...

class Adam(BaseOptimizer):     # 具体策略3
    def step(self, params, grads): ...
```

好处：`Trainer` 只知道"有一个优化器，它有 `step()` 方法"，不关心是哪种具体优化器。切换优化器只需改 `config.py` 的一行配置，训练代码完全不变。

---

## 12.2 SGD 实现

```python
class SGD(BaseOptimizer):
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate

    def step(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]
```

### 逐行解析

```python
for key in params:
    # key 依次是 'W1', 'b1', 'W2', 'b2'
    params[key] -= self.lr * grads[key]
    # params['W1'] -= 0.1 * grads['W1']
    # params['b1'] -= 0.1 * grads['b1']
    # ...
```

**`-=` 是原地更新（in-place）**：直接修改 `params` 字典中的数组，不创建新数组。这样 `model.params` 的内容被直接更新——因为 Python 中字典和 numpy 数组是引用传递。

**维度：** 每个 `params[key]` 和对应 `grads[key]` 形状相同，逐元素相减，结果形状不变。

---

## 12.3 Momentum 实现

```python
class Momentum(BaseOptimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self._velocity = {}      # 速度向量，首次调用时初始化

    def step(self, params, grads):
        # 首次调用：初始化速度（形状和参数相同，全零）
        if not self._velocity:
            for key in params:
                self._velocity[key] = np.zeros_like(params[key])

        for key in params:
            self._velocity[key] = (
                self.momentum * self._velocity[key]   # 保留历史速度
                - self.lr * grads[key]                # 当前梯度带来的加速
            )
            params[key] += self._velocity[key]        # 按速度更新参数
```

### 速度初始化

```python
if not self._velocity:   # 第一次调用时，self._velocity 是空字典 {}
    for key in params:
        self._velocity[key] = np.zeros_like(params[key])
        # np.zeros_like(W1) → (784, 50) 的全零矩阵
        # np.zeros_like(b1) → (50,) 的全零向量
```

**为什么用 `np.zeros_like`？** 它创建与参数形状、类型完全相同的全零数组，不需要手写 shape。

### 速度更新

```python
v ← μv - αg
θ ← θ + v
```

对应代码：
```python
self._velocity[key] = 0.9 * self._velocity[key] - self.lr * grads[key]
params[key] += self._velocity[key]
```

注意 `+=`（不是 `-=`）：因为速度 $v$ 本身已经是负方向（$-\alpha g$ 使速度朝梯度相反方向），所以参数加速度就好。

---

## 12.4 Adam 实现

```python
class Adam(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr    = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = epsilon
        self._t = 0     # 步数计数器（用于偏差修正）
        self._m = {}    # 一阶矩（梯度均值）
        self._v = {}    # 二阶矩（梯度平方均值）
```

### step 方法

```python
    def step(self, params, grads):
        # 初始化（首次调用）
        if not self._m:
            for key in params:
                self._m[key] = np.zeros_like(params[key])
                self._v[key] = np.zeros_like(params[key])

        self._t += 1   # 步数加 1（用于偏差修正）
```

**关键步骤：预计算有效学习率**

```python
        # lr_t = α · sqrt(1 - β₂ᵗ) / (1 - β₁ᵗ)
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self._t) \
                       / (1.0 - self.beta1 ** self._t)
```

这把偏差修正合并进学习率，避免在循环内重复计算，等价于：

```python
m_hat = m / (1 - beta1^t)     # 修正后的一阶矩
v_hat = v / (1 - beta2^t)     # 修正后的二阶矩
theta -= lr * m_hat / (sqrt(v_hat) + eps)

# 等价于：
# theta -= lr * sqrt(1-beta2^t) / (1-beta1^t) * m / (sqrt(v) + eps)
#       = lr_t * m / (sqrt(v) + eps')
```

**验证早期步骤（t=1）：**

```python
lr_t = 0.001 * sqrt(1 - 0.999^1) / (1 - 0.9^1)
     = 0.001 * sqrt(0.001) / 0.1
     = 0.001 * 0.0316 / 0.1
     = 0.000316
```

早期有效学习率比名义学习率 0.001 还小，因为偏差修正让初期的更新更保守——在矩估计还不准确时，步子迈小一点。

**参数更新：**

```python
        for key in params:
            g = grads[key]

            # 一阶矩更新：m ← β₁m + (1-β₁)g
            self._m[key] = self.beta1 * self._m[key] + (1.0 - self.beta1) * g

            # 二阶矩更新：v ← β₂v + (1-β₂)g²
            self._v[key] = self.beta2 * self._v[key] + (1.0 - self.beta2) * g ** 2

            # 参数更新：θ ← θ - lr_t · m / (√v + ε)
            params[key] -= lr_t * self._m[key] / (np.sqrt(self._v[key]) + self.eps)
```

**`g ** 2` 是逐元素平方**（不是矩阵乘法），结果 shape 和 `g` 相同。

**`np.sqrt(self._v[key])` 也是逐元素开方**，shape 不变。

---

## 12.5 工厂函数

```python
def build_optimizer(optimizer_name: str, learning_rate: float) -> BaseOptimizer:
    registry = {
        'sgd':      SGD,
        'momentum': Momentum,
        'adam':     Adam,
    }
    name = optimizer_name.lower()
    if name not in registry:
        raise ValueError(f"未知优化器: {optimizer_name!r}，可选: {list(registry.keys())}")
    return registry[name](learning_rate=learning_rate)
```

在 `main.py` 中调用：

```python
optimizer = build_optimizer(
    optimizer_name=TRAIN_CONFIG['optimizer'],    # 'adam'
    learning_rate=TRAIN_CONFIG['learning_rate'], # 0.001
)
# → Adam(learning_rate=0.001)
```

---

## 12.6 三种优化器的内存开销对比

| 优化器 | 存储的额外状态 | 额外内存 |
|--------|--------------|---------|
| SGD | 无 | 0 |
| Momentum | 速度向量 (v)，与参数同 shape | ×1 倍 |
| Adam | 一阶矩 (m) + 二阶矩 (v)，各与参数同 shape | ×2 倍 |

本项目参数总量约 40K，额外开销很小。但在大模型（数十亿参数）中，Adam 的额外内存是一个显著成本。

---

## 12.7 小结

| 优化器 | 额外状态 | 更新公式 | 推荐学习率 |
|--------|---------|---------|-----------|
| SGD | 无 | $\theta -= \alpha g$ | 0.1 |
| Momentum | 速度 $v$ | $v = \mu v - \alpha g$; $\theta += v$ | 0.01 |
| Adam | 矩 $m, v$；步数 $t$ | $\theta -= \text{lr\_t} \cdot m / (\sqrt{v}+\varepsilon)$ | 0.001 |

---

[← 第 11 章](chapter-11-model.md) | [返回目录](README.md) | [第 13 章：训练器 →](chapter-13-trainer.md)
