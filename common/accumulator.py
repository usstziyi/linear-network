import torch

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        # 初始化n个变量，每个变量的初始值为0.0
        self.data = [0.0] * n

    def add(self, *args):
        # 将输入的args中的每个元素转换为浮点数，并与当前数据进行累加
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        # 重置所有变量的值为0.0
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        # 返回第idx个变量的值
        return self.data[idx]