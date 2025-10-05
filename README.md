# Linear Network

一个基于PyTorch实现的线性模型库，包含线性回归和Softmax分类等基础机器学习模型。

## 项目概述

本项目实现了经典的线性模型，包括：
- **线性回归**：用于回归任务的单层神经网络
- **Softmax分类**：用于多分类任务的线性分类器
- **通用工具**：数据加载、动画可视化、损失函数等辅助工具

## 目录结构

```
linear-network/
├── classification/          # 分类模型
│   ├── __init__.py
│   └── softmax_classification.py  # Softmax分类模型
├── regression/              # 回归模型
│   ├── __init__.py
│   └── linear_regression.py       # 线性回归模型
├── common/                  # 通用工具
│   ├── __init__.py
│   ├── accumulator.py      # 累加器
│   ├── animator.py         # 动画可视化
│   ├── cpu_count.py        # CPU核心数
│   ├── cross_entropy.py    # 交叉熵损失
│   ├── normal.py           # 正态分布
│   ├── runtime.py          # 运行时工具
│   └── softmax.py          # Softmax函数
├── loss/                   # 损失函数
│   ├── MLE.md             # 最大似然估计文档
│   └── __init__.py
├── test_linear_regression.py      # 线性回归测试
├── test_softmax_classification.py # Softmax分类测试
└── README.md               # 项目说明
```

## 安装依赖

```bash
pip install torch torchvision matplotlib numpy
```

## 快速开始

### 线性回归

```python
from regression import LinearRegressionModel
from test_linear_regression import synthetic_data, load_array
import torch

# 生成数据
w = torch.tensor([2, -3.4])
b = torch.tensor(4.2)
X, y = synthetic_data(w, b, 1000)

# 创建模型
model = LinearRegressionModel()

# 训练模型
# 具体训练代码参考 test_linear_regression.py
```

### Softmax分类

```python
from classification import SoftmaxClassification

# 创建模型
model = SoftmaxClassification()

# 模型说明
# - 输入：图像数据 (B, 1, 28, 28)
# - 输出：10个类别的logits (B, 10)
# - 配合CrossEntropyLoss使用，内部自动应用softmax
```

## 模型说明

### LinearRegressionModel
- 单层线性回归模型
- 输入特征数：2
- 输出特征数：1
- 使用正态分布初始化权重

### SoftmaxClassification
- 线性分类器，用于多分类任务
- 自动展平输入数据
- 输出未归一化的logits
- 配合CrossEntropyLoss使用

## 工具模块

### Animator
实时可视化训练过程的动画工具：

```python
from common.animator import Animator

# 开启交互模式
Animator.enable_interactive()

# 创建动画器
animator = Animator(xlabel='epoch', ylabel='loss', legend=['train loss'])

# 在训练循环中更新
animator.add(epoch, train_loss)

# 关闭交互模式
Animator.disable_interactive()
```

### 其他工具
- `accumulator.py`：累加多个变量
- `cross_entropy.py`：交叉熵损失计算
- `normal.py`：正态分布相关
- `runtime.py`：运行时工具

## 运行测试

```bash
# 运行线性回归测试
python test_linear_regression.py

# 运行Softmax分类测试
python test_softmax_classification.py
```

## 技术特点

- **模块化设计**：每个模型和工具独立封装
- **PyTorch原生**：使用PyTorch标准接口
- **可视化支持**：内置动画可视化工具
- **易于扩展**：清晰的代码结构便于添加新模型

## 学习价值

本项目适合机器学习初学者学习：
- 线性模型的基本原理
- PyTorch模型构建流程
- 数据加载和预处理
- 训练循环和可视化
- 模型评估和测试

## 许可证

MIT License