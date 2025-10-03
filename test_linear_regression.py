import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from d2l import torch as d2l
from regression import LinearRegressionModel

# 生成数据集，带高斯正态分布的噪声
def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise.

    参数:
        w (torch.Tensor): 权重向量
        b (torch.Tensor): 偏置项
        num_examples (int): 生成样本的数量
    """
    # num_examples(B)
    # len(w) = D
    # X(B,D) 
    # 初始化X服从标准正态分布
    X = d2l.normal(mean=0, std=1, shape=(num_examples, len(w)))

    # 初始化y
    # X(B,D)
    # w(D)
    # b(1)
    # y(B,)
    y = d2l.matmul(X, w) + b
    # 初始化y,添加高斯正态分布噪声
    # y(B,)
    y += d2l.normal(mean=0, std=0.01, shape=y.shape)
    # y(B,)->(B,1)
    y = y.reshape(-1, 1)

    return X, y

# 加载数组数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = TensorDataset(*data_arrays)
    # 在 PyTorch 的 DataLoader 中，数据是在每个 epoch 开始时、生成 batch 之前被打乱的，
    # 具体发生在 DataLoader 内部创建数据索引顺序的环节。
    # 这有助于模型泛化，避免过拟合
    return DataLoader(dataset, batch_size, shuffle=is_train)





"""
(在高斯噪声假设下)
为什么最小化均方误差(MSE) = 最大化似然(MLE,Maximum Likelihood Estimation)

假设我们的模型输出为 y，真实标签为 t，噪声服从高斯分布 N(0, σ^2)。
则似然函数为：
L(θ) = Π p(t_i | y_i, θ)
= Π N(t_i | y_i, σ^2)
= Π exp(-0.5 * (t_i - y_i)^2 / σ^2) / √(2 * π * σ^2)
取负对数，得到均方误差：
MSE(θ) = -1/n * Σ (t_i - y_i)^2
= -1/n * Σ (t_i - w^T x_i - b)^2
= -1/n * Σ (t_i - ŷ_i)^2
其中 ŷ_i = w^T x_i + b 为模型的预测输出。
我们的目标是最小化均方误差，即找到最优的 w 和 b。
这等价于最大化似然函数，即找到使得 p(t | x, θ) 最大的 w 和 b。
"""
def train_Linear_Regression(model, data_iter, features, labels, num_epochs=3):
    # 定义损失函数：均方误差损失函数
    loss = nn.MSELoss() # 默认 reduction='mean'
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    # 训练模型epoch
    for epoch in range(num_epochs):
        # 训练模型batch
        for X, y in data_iter:
            # 1. 前向传播，创建新计算图，暂时不用梯度图
            # y_hat(B,1)
            y_hat = model(X)
            # 2. 损失传播，扩展计算图，暂时不用梯度图
            l = loss(y_hat, y) # 此时l是标量：0维张量
            # 3. 梯度清零，不修改计算图，会将梯度图设为0
            optimizer.zero_grad()
            # 4. 反向传播，依据计算图，会更新梯度图
            l.backward()
            # 5. 依据梯度图，原地更新参数w,b，不修改计算图，不修改梯度图
            optimizer.step()
            # 6. 循环结束，计算图会被销毁，梯度图会被在下一轮循环中被清零

        # 每个epoch结束,计算并打印损失
        # 自动微分系统（autograd）只在需要计算梯度时才记录操作并构建计算图
        with torch.no_grad():
            # 前向传播，此时不会创建新计算图，暂时不使用梯度图
            y_hat = model(features)
            # 计算损失，此时不会扩展计算图，暂时不使用梯度图
            l = loss(y_hat, labels)
        # 打印损失
        print(f'epoch {epoch + 1}, loss {l:f}')

def main():
    # true_w(D)
    true_w = torch.tensor([2, -3.4])
    # true_b(1)
    true_b = 4.2
    # 生成数据集
    # features(B,D)
    # labels(B,1)
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    # 加载数据集
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    # 创建模型
    model = LinearRegressionModel()
    # 训练模型
    train_Linear_Regression(model, data_iter, features, labels, num_epochs=10)
    # 打印模型参数
    w = model.linear.weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = model.linear.bias.data
    print('b的估计误差：', true_b - b)


    # 测试模型
    with torch.no_grad():
        # test_features(B,D)
        test_features = torch.normal(mean=0, std=1, size=(10, 2)) # 10个样本，每个样本2个特征，均值为0，标准差为1
        print('test_features:', test_features)
        # test_labels(B,1)
        test_labels = model(test_features)
        print('test_labels:', test_labels)

if __name__ == '__main__':
    main()



