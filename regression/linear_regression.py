"""
我们可以将线性回归模型视为仅由单个人工神经元组成的神经网络，或称为单层神经网络。
该模型通过单个仿射变换将我们的输入直接映射到输出,然后进行softmax操作(外部)。
"""


from torch import nn



# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义线性层,输入特征数为2,输出特征数为1
        self.linear = nn.Linear(2, 1)
        # 初始化权重和偏置,一般不做初始化,因为默认初始化已经足够好
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.fill_(0)
    
    def forward(self, x):
        # return(B,1)
        return self.linear(x)