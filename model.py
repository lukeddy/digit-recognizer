"""
model.py — 两层神经网络模型（含完整反向传播）
-----------------------------------------------
将网络结构、前向传播与反向传播封装为一个清晰的类。

网络结构
--------
  输入层  →  隐藏层（Sigmoid）  →  输出层（Softmax）
  (784)       (hidden_size)          (10)

已实现方法
----------
- forward()              : 前向传播，返回预测概率
- loss()                 : 计算交叉熵损失
- accuracy()             : 计算分类准确率
- gradient()             : 统一梯度接口，支持 'numerical' / 'backprop'
- _backprop_gradient()   : 解析梯度（反向传播，已实现）
- _numerical_gradient()  : 数值梯度（中心差分，用于验证）

反向传播推导（关键公式）
--------------------------
前向传播：
  a1 = X @ W1 + b1              (n, hidden)
  z1 = sigmoid(a1)              (n, hidden)
  a2 = z1 @ W2 + b2             (n, output)
  y  = softmax(a2)              (n, output)
  L  = cross_entropy(y, t)      scalar

反向传播（链式法则，从输出层向输入层逐层求偏导）：

  第二层（Softmax + Cross-Entropy 联合梯度，推导极简洁）：
    ∂L/∂a2 = (y - t_onehot) / n            (n, output)

    ∂L/∂W2 = z1^T  @  ∂L/∂a2             (hidden, output)
    ∂L/∂b2 = Σ_行 ∂L/∂a2                 (output,)
    ∂L/∂z1 = ∂L/∂a2  @  W2^T             (n, hidden)

  第一层（Sigmoid 导数传播）：
    ∂L/∂a1 = ∂L/∂z1  ×  sigmoid'(z1)     (n, hidden)
           = ∂L/∂z1  ×  z1 × (1 − z1)   （逐元素乘）

    ∂L/∂W1 = X^T  @  ∂L/∂a1             (input, hidden)
    ∂L/∂b1 = Σ_行 ∂L/∂a1                 (hidden,)
"""

import numpy as np
from functions import sigmoid, sigmoid_grad, relu, relu_grad, softmax, cross_entropy_error
from config import MODEL_CONFIG


class TwoLayerNet:
    """
    两层全连接神经网络（输入层 → 隐藏层 → 输出层）。

    参数
    ----
    input_size       : 输入特征维度（MNIST = 784）
    hidden_size      : 隐藏层节点数
    output_size      : 输出类别数（MNIST = 10）
    weight_init_std  : 权重初始化标准差（过大→梯度爆炸，过小→梯度消失）
    """

    def __init__(
        self,
        input_size: int  = MODEL_CONFIG['input_size'],
        hidden_size: int = MODEL_CONFIG['hidden_size'],
        output_size: int = MODEL_CONFIG['output_size'],
        weight_init_std: float = MODEL_CONFIG['weight_init_std'],
    ):
        # 参数字典（W1, b1, W2, b2）
        # 命名对应层次：
        #   W1 (input_size × hidden_size) — 第一层权重
        #   b1 (hidden_size,)             — 第一层偏置
        #   W2 (hidden_size × output_size)— 第二层权重
        #   b2 (output_size,)             — 第二层偏置
        self.params: dict = {
            'W1': np.random.randn(input_size, hidden_size) * weight_init_std,
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * weight_init_std,
            'b2': np.zeros(output_size),
        }

    # ---------------------------------------------------------------- #
    #  前向传播
    # ---------------------------------------------------------------- #
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播，返回各类别的预测概率分布。

        X : shape (n, input_size) — n 个样本
        返回 : shape (n, output_size) — 每个样本属于各类别的概率
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 第一层：线性变换 + Sigmoid 激活
        a1 = X @ W1 + b1        # (n, hidden_size)
        z1 = sigmoid(a1)        # (n, hidden_size)

        # 第二层：线性变换 + Softmax 输出概率
        a2 = z1 @ W2 + b2       # (n, output_size)
        y  = softmax(a2)        # (n, output_size)

        return y

    # ---------------------------------------------------------------- #
    #  损失计算
    # ---------------------------------------------------------------- #
    def loss(self, X: np.ndarray, t: np.ndarray) -> float:
        """
        计算交叉熵损失。

        X : shape (n, input_size)  — 输入特征
        t : shape (n,)             — 真实标签（整数编码）
        返回 : scalar 损失值
        """
        y = self.forward(X)
        return cross_entropy_error(y, t)

    # ---------------------------------------------------------------- #
    #  准确率计算
    # ---------------------------------------------------------------- #
    def accuracy(self, X: np.ndarray, t: np.ndarray) -> float:
        """
        计算分类准确率。

        X : shape (n, input_size)
        t : shape (n,) — 整数标签
        返回 : float，正确预测比例 [0, 1]
        """
        y_proba = self.forward(X)
        y_pred  = np.argmax(y_proba, axis=1)   # 取概率最大的类别
        return float(np.mean(y_pred == t))

    # ---------------------------------------------------------------- #
    #  梯度计算接口（当前：数值梯度；预留反向传播替换入口）
    # ---------------------------------------------------------------- #
    def gradient(self, X: np.ndarray, t: np.ndarray, method: str = 'numerical') -> dict:
        """
        计算所有参数的梯度，返回与 self.params 结构相同的字典。

        参数
        ----
        X      : 输入特征
        t      : 真实标签
        method : 梯度计算方式
                 - 'numerical'  数值梯度（当前实现，慢但无需推导）
                 - 'backprop'   反向传播（TODO：学习反向传播后实现）

        返回
        ----
        grads : dict，键与 self.params 相同，值为对应梯度数组
        """
        if method == 'numerical':
            return self._numerical_gradient(X, t)
        elif method == 'backprop':
            return self._backprop_gradient(X, t)
        else:
            raise ValueError(f"未知的梯度计算方法: {method!r}，可选 'numerical' / 'backprop'")

    def _numerical_gradient(self, X: np.ndarray, t: np.ndarray) -> dict:
        """数值梯度（中心差分法）——当前实现。"""
        from gradient import numerical_gradient as _num_grad

        # 把损失函数包装为只关于参数的函数
        loss_fn = lambda _: self.loss(X, t)

        grads = {}
        for key in self.params:
            grads[key] = _num_grad(loss_fn, self.params[key])
        return grads

    def _backprop_gradient(self, X: np.ndarray, t: np.ndarray) -> dict:
        """
        反向传播梯度（解析梯度，Backpropagation）。

        原理
        ----
        利用链式法则，从输出层向输入层逐层计算每个参数对损失的偏导数。
        比数值梯度快约 hidden_size 倍，是实际训练神经网络的标准做法。

        关键技巧：Softmax + Cross-Entropy 的联合梯度
        -----------------------------------------------
        单独推导 softmax 的 Jacobian 矩阵比较复杂（每个输出都依赖所有输入）。
        但当 softmax 与交叉熵组合使用时，联合梯度化简为极简洁的形式：

            ∂L/∂a2[i, j] = y[i, j] - 1_{j == t[i]}

        除以批量大小 n 后得到平均梯度，与损失函数的"平均"一致。
        直觉：模型对正确类别预测概率高时，梯度接近 0；预测错误时，梯度大。

        参数
        ----
        X : shape (n, input_size)  — 输入特征（当前 batch）
        t : shape (n,)             — 真实整数标签

        返回
        ----
        grads : dict，键 = {'W1', 'b1', 'W2', 'b2'}，值 = 对应梯度 ndarray
        """
        n = X.shape[0]
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # ── ① 前向传播（缓存中间值，反向传播需要用到）──────────── #
        a1 = X @ W1 + b1          # 第一层线性变换    (n, hidden)
        z1 = sigmoid(a1)          # 第一层 Sigmoid 激活 (n, hidden)
        a2 = z1 @ W2 + b2         # 第二层线性变换    (n, output)
        y  = softmax(a2)          # 第二层 Softmax 输出 (n, output)

        # ── ② 输出层反向：Softmax + Cross-Entropy 联合梯度 ───────── #
        # 构造 one-hot 矩阵：将整数标签 t 转为概率矩阵格式
        dy = y.copy()             # shape (n, output)
        dy[np.arange(n), t] -= 1  # 正确类别位置减 1（对应公式 y - 1_{j==t[i]}）
        dy /= n                   # 除以 n 得到平均梯度（对应损失函数中的 /n）

        # W2、b2 的梯度：z1 是 a2 的"输入"
        dW2 = z1.T @ dy           # (hidden, n) @ (n, output) → (hidden, output)
        db2 = np.sum(dy, axis=0)  # 对 n 个样本求和           → (output,)

        # ── ③ 将梯度传回隐藏层 ────────────────────────────────────── #
        # dy @ W2.T：将输出层梯度"反向流过"第二层权重矩阵
        dz1 = dy @ W2.T           # (n, output) @ (output, hidden) → (n, hidden)

        # ── ④ 隐藏层反向：Sigmoid 导数 ───────────────────────────── #
        # sigmoid'(a1) = z1 * (1 - z1)（直接用已算好的 z1，避免重复计算）
        da1 = dz1 * sigmoid_grad(z1)  # 逐元素乘    (n, hidden)

        # W1、b1 的梯度：X 是 a1 的"输入"
        dW1 = X.T @ da1           # (input, n) @ (n, hidden) → (input, hidden)
        db1 = np.sum(da1, axis=0) # 对 n 个样本求和           → (hidden,)

        # ── ⑤ 整理并返回梯度字典 ─────────────────────────────────── #
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
