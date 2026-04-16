"""
optimizer.py — 参数更新策略（优化器）
--------------------------------------
遵循"策略模式"：所有优化器继承抽象基类 BaseOptimizer，
只需实现 step() 方法，Trainer 无需关心具体更新逻辑。

已实现
------
- SGD      : 随机梯度下降（最基础）
- Momentum : 动量法（引入速度项，加速收敛，减小震荡）
- Adam     : 自适应矩估计（结合 Momentum + 自适应学习率，目前最常用）

各优化器对比
------------
  SGD       更新规则最简单，但收敛慢、对学习率敏感
  Momentum  引入"速度"，像小球滚下山坡，方向一致时加速，震荡时减弱
  Adam      自动为每个参数调整学习率，实践中收敛最稳定

工厂函数
--------
  build_optimizer(name, lr) → 根据名称返回对应优化器实例
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    """优化器抽象基类。"""

    @abstractmethod
    def step(self, params: dict, grads: dict) -> None:
        """
        原地更新 params（in-place）。

        params : 模型参数字典 {'W1': ..., 'b1': ..., ...}
        grads  : 对应梯度字典，结构与 params 相同
        """
        ...


# ------------------------------------------------------------------ #
#  SGD — 随机梯度下降
# ------------------------------------------------------------------ #
class SGD(BaseOptimizer):
    """
    随机梯度下降（Stochastic Gradient Descent）。

    更新规则：θ ← θ - lr × ∇θ L
    """

    def __init__(self, learning_rate: float = 0.1):
        self.lr = learning_rate

    def step(self, params: dict, grads: dict) -> None:
        for key in params:
            params[key] -= self.lr * grads[key]


# ------------------------------------------------------------------ #
#  Momentum — 动量法
# ------------------------------------------------------------------ #
class Momentum(BaseOptimizer):
    """
    动量法优化器（SGD with Momentum）。

    原理
    ----
    引入速度向量 v，模拟物理中"带动量的小球"滚下损失曲面：
      - 梯度方向一致时，速度不断累积 → 收敛更快
      - 梯度方向振荡时，速度相互抵消 → 减弱震荡

    更新规则
    --------
        v ← momentum × v  −  lr × ∇θ L
        θ ← θ + v

    与 SGD 对比
    -----------
    SGD 每步只看当前梯度；Momentum 额外记忆了历史梯度方向，
    能在"沟壑形"损失曲面上沿长轴方向加速，减少横向震荡。

    参数
    ----
    learning_rate : 学习率（默认 0.01，通常比 SGD 的 0.1 小）
    momentum      : 速度衰减系数（默认 0.9，即保留 90% 历史速度）
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self._velocity: dict = {}   # 速度向量，首次 step 时按参数形状初始化

    def step(self, params: dict, grads: dict) -> None:
        # 首次调用：为每个参数初始化速度向量（全零，静止出发）
        if not self._velocity:
            for key in params:
                self._velocity[key] = np.zeros_like(params[key])

        for key in params:
            # 速度更新：动量加速 + 梯度减速
            self._velocity[key] = (
                self.momentum * self._velocity[key]
                - self.lr * grads[key]
            )
            # 参数更新：沿速度方向前进
            params[key] += self._velocity[key]


# ------------------------------------------------------------------ #
#  Adam — 自适应矩估计
# ------------------------------------------------------------------ #
class Adam(BaseOptimizer):
    """
    Adam 优化器（Adaptive Moment Estimation）。

    原理
    ----
    同时维护梯度的**一阶矩（均值）**和**二阶矩（方差）**，
    为每个参数自适应地调整有效学习率：
      - 梯度稳定的参数 → 二阶矩大 → 有效学习率小（防止过冲）
      - 梯度稀疏的参数 → 二阶矩小 → 有效学习率大（加速学习）

    更新规则
    --------
    （t 为当前步数，用于偏差修正）

        m ← β₁·m  +  (1 − β₁)·∇θ L          # 一阶矩：梯度的指数移动平均
        v ← β₂·v  +  (1 − β₂)·(∇θ L)²       # 二阶矩：梯度平方的指数移动平均

        m̂ = m / (1 − β₁^t)                   # 偏差修正（初始阶段矩估计偏低）
        v̂ = v / (1 − β₂^t)

        θ ← θ  −  lr · m̂ / (√v̂ + ε)         # ε 防止除零

    超参数默认值（论文推荐）
    -----------------------
    β₁ = 0.9    一阶矩衰减率
    β₂ = 0.999  二阶矩衰减率（变化更慢，追踪梯度幅度的长期均值）
    ε  = 1e-8   数值稳定项
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.lr    = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = epsilon
        self._t: int  = 0           # 全局步数计数（用于偏差修正）
        self._m: dict = {}          # 一阶矩向量（梯度均值）
        self._v: dict = {}          # 二阶矩向量（梯度平方均值）

    def step(self, params: dict, grads: dict) -> None:
        # 首次调用：初始化一、二阶矩向量（全零）
        if not self._m:
            for key in params:
                self._m[key] = np.zeros_like(params[key])
                self._v[key] = np.zeros_like(params[key])

        self._t += 1  # 步数加一

        # 预计算偏差修正后的有效学习率（避免在循环内重复计算）
        # lr_t 在训练初期较大（修正偏低的矩估计），后期趋向 lr
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self._t) / (1.0 - self.beta1 ** self._t)

        for key in params:
            g = grads[key]

            # 更新一阶矩（梯度的指数移动平均）
            self._m[key] = self.beta1 * self._m[key] + (1.0 - self.beta1) * g

            # 更新二阶矩（梯度平方的指数移动平均）
            self._v[key] = self.beta2 * self._v[key] + (1.0 - self.beta2) * g ** 2

            # 参数更新：lr_t 已含偏差修正，此处直接用原始矩向量
            params[key] -= lr_t * self._m[key] / (np.sqrt(self._v[key]) + self.eps)


# ------------------------------------------------------------------ #
#  工厂函数
# ------------------------------------------------------------------ #
def build_optimizer(optimizer_name: str, learning_rate: float) -> BaseOptimizer:
    """根据配置名称创建对应优化器实例。"""
    registry = {
        'sgd':      SGD,
        'momentum': Momentum,
        'adam':     Adam,
    }
    name = optimizer_name.lower()
    if name not in registry:
        raise ValueError(f"未知优化器: {optimizer_name!r}，可选: {list(registry.keys())}")
    return registry[name](learning_rate=learning_rate)
