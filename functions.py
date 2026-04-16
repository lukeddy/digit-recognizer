"""
functions.py — 激活函数、损失函数及其导数
-------------------------------------------
涵盖神经网络前向传播与反向传播所需的全部基础函数。

前向传播用函数
  sigmoid, relu, softmax, identity

反向传播用导数函数（grad）
  sigmoid_grad, relu_grad
  注：softmax + cross_entropy 组合梯度在 model.py 中直接计算，
      公式为 (y - t_onehot) / n，无需单独封装。

损失函数
  cross_entropy_error — 多分类标准损失（配合 softmax 输出层）
  mean_squared_error  — 回归 / 演示用（一般不与 softmax 搭配）
"""

import numpy as np


# ================================================================== #
#  激活函数（前向传播）
# ================================================================== #

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid 激活函数：将任意实数映射到 (0, 1)。

    公式：σ(x) = 1 / (1 + e^{-x})
    常用于隐藏层（现代网络多用 ReLU），以及二分类输出层。
    缺点：深层网络容易发生梯度消失（输出饱和区梯度趋近 0）。
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid 函数的导数（反向传播用）。

    公式：σ'(x) = σ(x) · (1 − σ(x))

    参数
    ----
    z : sigmoid 的**输出值**，即 z = sigmoid(x)
        直接传入已算好的 z，避免重复计算。

    推导
    ----
    设 y = σ(x)，则 dy/dx = y · (1 − y)
    在反向传播中：δ_{前层} = δ_{后层} · σ'(x) = δ_{后层} · z · (1 − z)
    """
    return z * (1.0 - z)


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU 激活函数：保留正值，负值置零。

    公式：ReLU(x) = max(0, x)
    优点：计算简单、正区间梯度恒为 1（缓解梯度消失）。
    缺点：负区间梯度为 0，可能导致"神经元死亡"（Dead ReLU）。
    """
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """
    ReLU 函数的导数（反向传播用）。

    公式：ReLU'(x) = 1 if x > 0 else 0

    参数
    ----
    x : ReLU 的**输入值**（线性激活前的值 a），不是输出。

    推导
    ----
    ReLU 对 x > 0 的梯度为 1，x ≤ 0 的梯度为 0。
    即：梯度信号仅在"正向激活"的节点上流过。
    """
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax 函数：将向量转换为概率分布（各分量之和为 1）。

    公式：softmax(x_i) = e^{x_i} / Σ e^{x_j}
    支持 1D（单样本）和 2D（批量）输入，内置溢出防护。

    溢出防护
    --------
    减去各样本的最大值不改变 softmax 输出：
        softmax(x) = softmax(x - max(x))
    这样所有指数的底数 ≤ 1，避免 e^{很大数} 溢出。
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def identity(x: np.ndarray) -> np.ndarray:
    """恒等函数，用于回归任务的输出层（直接输出 a2）。"""
    return x


# ================================================================== #
#  损失函数
# ================================================================== #

def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """
    交叉熵损失（Cross Entropy Error）。

    公式：L = -1/n · Σ_i log(y[i, t[i]])
    与 softmax 输出层配合使用时，其组合梯度极为简洁：
        ∂L/∂a2 = (y - t_onehot) / n
    （见 model.py 反向传播推导）

    参数
    ----
    y : 预测概率，shape (n, num_classes)，由 softmax 输出
    t : 真实标签，shape (n,)（整数标签）或 (n, num_classes)（one-hot）

    返回
    ----
    scalar — 平均交叉熵损失（除以样本数 n）
    """


    # 本段代码解释：https://gemini.google.com/share/b79303ee399f
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
    # 若 t 是 one-hot 编码，转换为整数标签
    if t.size == y.size:
        t = np.argmax(t, axis=1)
    n = y.shape[0]
    # 取正确类别的预测概率，+1e-7 防止 log(0) → -∞
    return -np.sum(np.log(y[np.arange(n), t] + 1e-7)) / n


def mean_squared_error(y: np.ndarray, t: np.ndarray) -> float:
    """
    均方误差损失（Mean Squared Error）。

    公式：L = 0.5 · Σ (y_i - t_i)²
    常用于回归任务；分类任务中配合 softmax 效果不如交叉熵。

    参数
    ----
    y : 网络输出，任意 shape
    t : 目标值，与 y 相同 shape

    返回
    ----
    scalar — 均方误差（乘以 0.5 使导数形式更简洁）
    """
    return 0.5 * np.sum((y - t) ** 2)
