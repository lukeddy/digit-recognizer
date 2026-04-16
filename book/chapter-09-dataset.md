# 第 9 章：数据加载与处理（dataset.py）

## 9.1 MNIST 数据集是什么？

MNIST（Modified National Institute of Standards and Technology）是深度学习领域最经典的入门数据集：

- **内容：** 手写数字 0-9 的灰度图像
- **格式：** 每张图 28×28 像素，像素值 0-255（0=白，255=黑）
- **规模：** 本项目使用 Kaggle 版本的 42,000 张训练图片

**CSV 文件格式（train.csv）：**

```
label, pixel0, pixel1, pixel2, ..., pixel783
1,     0,      0,      0,      ..., 0
0,     0,      0,      0,      ..., 0
4,     0,      0,      255,    ..., 128
...
```

- 第一列 `label`：图片对应的数字（0-9）
- 其余 784 列：28×28 图片的每个像素值（从左到右、从上到下逐行排列）

---

## 9.2 load_mnist 函数逐行解析

```python
def load_mnist(
    data_path=DATA_PATH,
    test_size=0.3,         # 30% 的数据作为测试集
    random_state=42,       # 固定随机种子，保证每次结果一样
    normalize=True,        # 是否归一化到 [0, 1]
):
```

### 步骤 1：读取 CSV

```python
data = pd.read_csv(data_path)
```

`data` 是一个 DataFrame，shape = `(42000, 785)`：
- 42,000 行：42,000 张图片
- 785 列：1 列标签 + 784 列像素

### 步骤 2：分离特征与标签

```python
X = data.drop(columns='label').values.astype(np.float64)
y = data['label'].values
```

- `X`：shape = `(42000, 784)`，dtype = float64（像素值转为浮点数，方便后续归一化）
- `y`：shape = `(42000,)`，dtype = int64（标签：0-9）

**为什么要用 float64？**

像素值原始是整数（0-255）。归一化时要做除法，整数除法在某些情况会截断小数，转为浮点数更安全。

### 步骤 3：划分训练集/测试集

```python
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

- 70% 训练：x_train shape = `(29400, 784)`，y_train shape = `(29400,)`
- 30% 测试：x_test shape = `(12600, 784)`，y_test shape = `(12600,)`

**为什么要分训练/测试集？**

训练集用于更新参数，测试集用于评估模型在**从未见过的数据**上的表现。如果只用训练集评估，模型可能只是"死记硬背"而不是真正学会规律——这叫**过拟合**。

**random_state=42 的作用：**

`train_test_split` 在划分时会随机打乱数据。固定随机种子意味着每次运行的划分结果完全一样，实验可复现。42 只是一个常见的习惯用法（来自"生命、宇宙以及一切的终极答案"）。

### 步骤 4：归一化（MinMaxScaler）

```python
if normalize:
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)
```

**MinMaxScaler 的作用：**

把每个特征（像素）的值从 [0, 255] 缩放到 [0, 1]：

$$x_{\text{归一化}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

对 MNIST 来说：$x_{\min} = 0$，$x_{\max} = 255$，所以：$x_{\text{归一化}} = \frac{x}{255}$

**为什么要归一化？**

1. Sigmoid 函数在输入很大时梯度趋近于 0（梯度消失）。把输入压缩到 [0,1]，Sigmoid 在有效区间内工作
2. 不同特征的量级相差太大时，梯度更新会不均匀，训练不稳定
3. 归一化后，权重初始化的标准差 0.01 与输入量级匹配，训练更容易

**为什么是 fit_transform + transform，而不是两次 fit_transform？**

```python
scaler.fit_transform(x_train)   # 在训练集上学习最大/最小值，然后变换
scaler.transform(x_test)        # 用训练集的最大/最小值变换测试集（不重新学习）
```

如果对测试集重新 fit，测试集的最大/最小值会被引入，这叫**数据泄露**（Data Leakage）：训练时"偷看"了测试集的统计信息，导致测试准确率虚高，不能反映真实性能。

---

## 9.3 数据维度总结

```
原始数据:
  CSV:  (42000, 785)  = 42000 张图片 × (1标签 + 784像素)

分离后:
  X:    (42000, 784)  每行是一张展平的图片
  y:    (42000,)      每个元素是对应图片的数字标签

划分后:
  x_train: (29400, 784)  ← 70%
  y_train: (29400,)
  x_test:  (12600, 784)  ← 30%
  y_test:  (12600,)

归一化后:
  值范围从 [0, 255] → [0.0, 1.0]，shape 不变
```

---

## 9.4 describe_dataset 函数

```python
def describe_dataset(x_train, x_test, y_train, y_test):
    print(f"训练集特征: {x_train.shape}  标签: {y_train.shape}")
    print(f"测试集特征: {x_test.shape}   标签: {y_test.shape}")
    print(f"特征值范围: [{x_train.min():.4f}, {x_train.max():.4f}]")
    print(f"类别分布:   {dict(zip(*np.unique(y_train, return_counts=True)))}")
```

运行后输出类似：
```
训练集特征: (29400, 784)  标签: (29400,)
测试集特征: (12600, 784)  标签: (12600,)
特征值范围: [0.0000, 1.0000]
类别分布: {0: 2908, 1: 3346, 2: 2900, 3: 3019, ...}
```

"类别分布"告诉我们每个数字出现了多少次，用于检查数据是否均衡（本数据集各类别数量大致相当）。

---

[← 第 8 章](chapter-08-project-overview.md) | [返回目录](README.md) | [第 10 章：激活与损失函数实现 →](chapter-10-functions.md)
