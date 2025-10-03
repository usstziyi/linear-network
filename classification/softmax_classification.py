
from torch import nn


"""
⚠️ 注意：虽然类名叫 SoftmaxClassification,但代码中并没有显式使用 nn.Softmax。这是因为:
在训练时，通常配合 nn.CrossEntropyLoss 使用，它内部会自动对 logits 应用 softmax 并计算交叉熵损失。
显式加 softmax 反而可能导致数值不稳定或重复计算。

这个模型虽然简单，但清晰展示了分类任务的基本流程：
输入 → 特征展平 → 线性映射 → 输出 logits →(外部) Softmax + 损失计算。
"""

class SoftmaxClassification(nn.Module):
    def __init__(self):
        super().__init__()
        # 展平层：将输入张量（如图像）展平成一维向量
        self.flatten = nn.Flatten()
        # 全连接层：将展平后的向量映射到输出类别数（10个类别）
        self.linear = nn.Linear(784, 10)

    def forward(self, X):
        # 展平输入张量X(B, 1, 28, 28) -> X(B, 784)
        X = self.flatten(X)
        # 全连接层映射到输出类别数（10个类别）
        # 这个线性层的输出通常被称为logits（未归一化的分类得分）
        # X(B, 784) -> X(B, 10)
        return self.linear(X)