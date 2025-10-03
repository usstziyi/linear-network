import torch
from softmax import Softmax


# y_hat(B,G)
# y(B,1)
def cross_entropy(y_hat, y):
    # y_hat(B,G)
    # y(B,1)
    # Pc(B,1)
    Pc= y_hat[range(len(y_hat)), y]
     # return(B,1)
    return - torch.log(Pc)

def main():
    # y_hat(B,G)
    y_hat = torch.normal(0, 1, (2, 5))
    print(y_hat)
    # y_hat(B,G)
    y_hat = Softmax(y_hat) # 对每个样本的预测值进行 softmax 归一化
    print(y_hat)
    # y(B,)
    y = torch.tensor([0, 3])
    # y_hat(B,1)
    # y(B)
    # loss(B)
    loss = cross_entropy(y_hat, y)
    print(loss)
    # loss(B,1)
    loss = loss.mean()
    print(loss)

if __name__ == '__main__':
    main()
