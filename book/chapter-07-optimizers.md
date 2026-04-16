# 第 7 章：参数更新与优化器

> 有了梯度，下一步是决定"如何用梯度更新参数"。不同的策略（优化器）对训练速度和最终效果影响巨大。

## 7.1 梯度下降的基本思想

想象你站在一座山上，目标是走到最低点（损失最小）。你的策略：

**每次朝当前位置最陡的下坡方向走一步**。

- 梯度 = 坡度（指向上坡方向）
- 负梯度 = 下坡方向
- 步长（学习率）= 每步走多远

```
    损失
     ↑
     |   /\
     |  /  \
     | /    \
     |/      \___
     |            \___
     +------------------→ 参数值
              ↑
          当前位置
          梯度<0（左侧是上坡）
          所以应该向右走（增大参数）
```

**基本更新规则：**

$$\theta \leftarrow \theta - \alpha \cdot \frac{\partial L}{\partial \theta}$$

其中 $\alpha$ 是**学习率**（Learning Rate），控制每步走多远。

---

## 7.2 SGD：随机梯度下降

### 7.2.1 公式

$$\boxed{\theta \leftarrow \theta - \alpha \cdot g}$$

其中 $g = \frac{\partial L}{\partial \theta}$ 是当前 batch 算出的梯度。

### 7.2.2 代码

```python
class SGD:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate

    def step(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]
```

就这么简单：参数 = 参数 - 学习率 × 梯度。

### 7.2.3 问题

SGD 的问题在某些损失曲面上很明显：

```
等高线图（椭圆形损失曲面）：

     ↑ 参数 W₂
     |  ___
     | /   \
     |      |
     |   ×  |  ← 最优点
     |      |
     | \___/
     |
     +----------→ 参数 W₁

SGD 的路径（锯齿状）：

     ↑         ← 在 W₂ 方向震荡
     | ↗↘↗↘↗
     |          → 在 W₁ 方向缓慢前进
```

在梯度大的方向（短轴），SGD 来回震荡；在梯度小的方向（长轴），前进缓慢。

---

## 7.3 Momentum：动量法

### 7.3.1 物理比喻

想象把小球放在损失曲面上，让它自由滚动：

- 在陡峭方向，小球快速加速，但在震荡方向，来回的力相互抵消
- 小球会沿着"河谷"加速滑向最低点

**Momentum 给梯度下降加入了"惯性"**。

### 7.3.2 公式

$$\boxed{v \leftarrow \mu \cdot v - \alpha \cdot g}$$
$$\boxed{\theta \leftarrow \theta + v}$$

其中：
- $v$：速度向量（初始为 0），记录"历史运动方向"
- $\mu$：动量系数（通常 0.9），保留多少历史速度
- $g$：当前梯度

### 7.3.3 直觉

展开速度 $v$ 的更新：

```
第 1 步：v₁ = -α·g₁
第 2 步：v₂ = μ·v₁ - α·g₂ = -μα·g₁ - α·g₂
第 3 步：v₃ = μ·v₂ - α·g₃ = -μ²α·g₁ - μα·g₂ - α·g₃
...
第 t 步：vₜ = -α·Σₖ μ^(t-k) · gₖ
```

当前速度 = **历史所有梯度的指数加权和**（越近的梯度权重越大）。

**效果：**
- 如果多次梯度方向一致 → 速度累积，加速前进
- 如果梯度方向来回震荡 → 速度相互抵消，减弱震荡

```python
class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self._velocity = {}

    def step(self, params, grads):
        if not self._velocity:
            for key in params:
                self._velocity[key] = np.zeros_like(params[key])

        for key in params:
            # 速度 = 保留历史速度 + 当前梯度带来的加速
            self._velocity[key] = self.momentum * self._velocity[key] - self.lr * grads[key]
            # 参数按速度方向更新
            params[key] += self._velocity[key]
```

### 7.3.4 与 SGD 对比

```
Momentum 的路径（比 SGD 平滑得多）：

     ↑         ← 震荡被抑制
     | →→→→→
     |          → 在主方向加速前进
```

---

## 7.4 Adam：自适应矩估计

### 7.4.1 设计思路

Adam 结合了两个思想：
1. **Momentum 的思路**：用历史梯度的均值（一阶矩）代替原始梯度
2. **AdaGrad 的思路**：根据梯度的历史方差（二阶矩）自动调整每个参数的学习率

**关键洞察：** 不同参数的梯度幅度差异很大。对梯度大的参数用小学习率，对梯度小的参数用大学习率——这样所有参数都能以合适的速度更新。

### 7.4.2 公式

每步 $t$：

**一阶矩（梯度的指数移动均值）：**
$$m \leftarrow \beta_1 \cdot m + (1 - \beta_1) \cdot g$$

**二阶矩（梯度平方的指数移动均值）：**
$$v \leftarrow \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$$

**偏差修正（修正初始阶段偏低的估计）：**
$$\hat{m} = \frac{m}{1 - \beta_1^t}, \quad \hat{v} = \frac{v}{1 - \beta_2^t}$$

**参数更新：**
$$\boxed{\theta \leftarrow \theta - \alpha \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}}$$

默认超参数（论文推荐值）：
- $\beta_1 = 0.9$（一阶矩衰减）
- $\beta_2 = 0.999$（二阶矩衰减）
- $\varepsilon = 10^{-8}$（防止除零）
- $\alpha = 0.001$（学习率）

### 7.4.3 为什么需要偏差修正？

初始时 $m = 0$，$v = 0$。

第一步更新后：
$$m = (1 - 0.9) \cdot g_1 = 0.1 \cdot g_1$$

但真正的梯度均值估计应该接近 $g_1$，而不是 $0.1 \cdot g_1$。初始几步的估计严重偏低。

偏差修正：$\hat{m} = \frac{m}{1 - 0.9^1} = \frac{0.1g_1}{0.1} = g_1$ ✓

随着训练步数 $t$ 增大，$\beta^t \rightarrow 0$，修正系数趋向 1，偏差修正逐渐失效（因为不再需要了）。

### 7.4.4 有效学习率的意义

Adam 的实际更新量（对单个参数）：

$$\Delta\theta = -\alpha \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}$$

用 $g$ 估计 $\hat{m}$，用 $g^2$ 估计 $\hat{v}$（极度简化）：

$$\approx -\alpha \cdot \frac{g}{\sqrt{g^2}} = -\alpha \cdot \text{sign}(g)$$

**Adam 的更新步长大约是固定的 $\alpha$！**（不管梯度多大多小）

但实际上用历史均值（而非瞬时值），所以比这更平滑——梯度一致的参数学习率大，震荡的参数学习率小。

### 7.4.5 代码实现

```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr    = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = epsilon
        self._t = 0
        self._m = {}   # 一阶矩
        self._v = {}   # 二阶矩

    def step(self, params, grads):
        if not self._m:
            for key in params:
                self._m[key] = np.zeros_like(params[key])
                self._v[key] = np.zeros_like(params[key])

        self._t += 1

        # 预计算偏差修正后的有效学习率
        # lr_t = α · sqrt(1-β₂ᵗ) / (1-β₁ᵗ)
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self._t) / (1.0 - self.beta1 ** self._t)

        for key in params:
            g = grads[key]
            self._m[key] = self.beta1 * self._m[key] + (1.0 - self.beta1) * g        # 一阶矩
            self._v[key] = self.beta2 * self._v[key] + (1.0 - self.beta2) * g ** 2  # 二阶矩
            params[key] -= lr_t * self._m[key] / (np.sqrt(self._v[key]) + self.eps)
```

注意代码中把偏差修正合并进了学习率 `lr_t`，而不是单独修正 m 和 v：

$$\text{lr\_t} = \alpha \cdot \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$$

然后直接用原始的 $m$ 和 $v$（未修正的）：

$$\theta \leftarrow \theta - \text{lr\_t} \cdot \frac{m}{\sqrt{v} + \varepsilon}$$

数学上等价，但只需一次修正计算，速度更快。

---

## 7.5 三种优化器对比

| 特性 | SGD | Momentum | Adam |
|------|-----|----------|------|
| 更新规则 | $\theta -= \alpha g$ | 引入速度 $v$ | 引入一、二阶矩 |
| 参数自适应 | 无 | 无 | 有（每参数独立学习率） |
| 收敛速度 | 慢 | 较快 | 最快（一般情况）|
| 调参难度 | 简单（只有 $\alpha$）| 较简单 | 相对复杂（但默认值通常够用）|
| 推荐学习率 | 0.01 ~ 0.1 | 0.01 | 0.001 |
| 适用场景 | 教学、简单问题 | 经典深度网络 | 大多数情况首选 |

---

## 7.6 学习率的影响

学习率 $\alpha$ 是最重要的超参数：

```
损失
↑
|  α 太大：在最优点附近来回跳跃，无法收敛
|    ↗↘↗↘↗
|
|  α 合适：平滑收敛到最优点
|    ↘___
|
|  α 太小：收敛极慢，可能困在局部极小值
|    ↘
|     ↘
|      ↘（很慢）
+------------------→ 训练步数
```

**本项目的建议：**
- SGD：lr = 0.1
- Momentum：lr = 0.01
- Adam：lr = 0.001（我们的配置）

---

## 7.7 训练循环总结

把反向传播（第 6 章）和参数更新（本章）合并，完整的一次训练步骤是：

```
① 取一个 mini-batch：x_batch, t_batch
② 前向传播：y = model.forward(x_batch)
③ 计算损失：L = cross_entropy(y, t_batch)
④ 反向传播：grads = model.backprop(x_batch, t_batch)
⑤ 参数更新：for θ in params: θ -= lr * grads[θ]  （SGD 情况）
⑥ 重复 ①-⑤，直到收敛
```

---

## 7.8 小结

| 概念 | 关键点 |
|------|--------|
| 梯度下降 | 沿负梯度方向更新：$\theta -= \alpha \cdot g$ |
| SGD | 最简单，但在椭圆损失曲面上震荡 |
| Momentum | 引入速度项，历史梯度加权，减少震荡 |
| Adam | 自适应学习率，一阶矩 + 二阶矩，实践中最稳定 |

> **第一部分结束！** 你现在已经有了理解这个神经网络项目所需的全部理论基础。接下来，我们进入第二部分：把这些理论翻译成代码。

---

[← 第 6 章](chapter-06-backpropagation.md) | [返回目录](README.md) | [第 8 章：项目结构 →](chapter-08-project-overview.md)
