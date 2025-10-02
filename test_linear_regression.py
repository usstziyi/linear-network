import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 生成数据集，带正态分布的噪声
def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise.

    参数:
        w (torch.Tensor): 权重向量
        b (torch.Tensor): 偏置项
        num_examples (int): 生成样本的数量
    """
    # num_examples(B)
    # X(B,D)
    X = d2l.normal(mean=0, std=1, shape=(num_examples, len(w)))
    
    y = d2l.matmul(X, w) + b
   
    y += d2l.normal(mean=0, std=0.01, shape=y.shape)
    # X(B,D)

    return X, d2l.reshape(y, shape=(-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)


# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)