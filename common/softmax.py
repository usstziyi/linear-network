import torch
# 实现 softmax 函数
def Softmax(X):
    # X(B,G)
    # X_exp(B,G)
    X_exp = torch.exp(X)
    # partition(B,1)
    partition = X_exp.sum(1, keepdim=True)
    # return(B,G)
    return X_exp / partition 


def main():
    X = torch.normal(0, 1, (2, 5))
    X_prob = Softmax(X)
    print(X_prob)
    print(X_prob.sum(1))

if __name__ == '__main__':
    main()