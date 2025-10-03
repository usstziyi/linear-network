import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from common import get_dataloader_workers
from classification import SoftmaxClassification
from common import Timer



# 加载Fashion-MNIST数据集
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        # PIL Image ──[Resize]──► PIL Image ──[ToTensor]──► Tensor
        # 将变换列表 trans（如 [Resize, ToTensor]）封装成一个可调用的组合变换对象
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans) # 组合变换对象, 用于将多个变换组合在一起
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))






# 训练Softmax分类模型
def train_SoftMaxClassification(model, train_iter, test_iter, num_epochs=10):
    # 定义损失函数：交叉熵损失函数
    # 在训练时，通常配合 nn.CrossEntropyLoss 使用，它内部会自动对 logits 应用 softmax 并计算交叉熵损失。
    loss = nn.CrossEntropyLoss(reduction='none')
    # 定义优化器：随机梯度下降（SGD）优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # 初始化计时器
    timer = Timer()

    # 训练模型
    for epoch in range(num_epochs):
        timer.start()  # 开始计时
        # 训练模型并计算训练损失和训练准确率
        train_metrics = d2l.Accumulator(3)
        model.train()
        for X, y in train_iter:
            y_hat = model(X) # 前向传播, 计算模型输出 logits
            l = loss(y_hat, y) # 内部会自动对 logits 应用 softmax 并计算交叉熵损失
            optimizer.zero_grad()
            l.mean().backward()
            optimizer.step()
            train_metrics.add(l.sum(), d2l.accuracy(y_hat, y), y.numel())
        train_loss, train_acc, train_samples = train_metrics
        # 评估模型
        model.eval()
        with torch.no_grad():
            test_metrics = d2l.Accumulator(3)
            for X, y in test_iter:
                y_hat = model(X) # 前向传播, 计算模型输出 logits
                l = loss(y_hat, y) # 内部会自动对 logits 应用 softmax 并计算交叉熵损失
                test_metrics.add(l.sum(), d2l.accuracy(y_hat, y), y.numel())
            test_loss, test_acc, test_samples = test_metrics
            timer.stop()  # 停止计时
            print(f"epoch {(epoch+1):2d}/{num_epochs}: train loss {train_loss/train_samples:.4f}, "
                  f"train acc {train_acc/train_samples:.4f}, "
                  f"test loss {test_loss/test_samples:.4f}, "
                  f"test acc {test_acc/test_samples:.4f}, "
                  f"time {timer.sum():.2f} sec")
    

def main():
    batch_size = 256
    num_epochs = 20
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    model = SoftmaxClassification()
    train_SoftMaxClassification(model, train_iter, test_iter, num_epochs)

if __name__ == "__main__":
    main()
