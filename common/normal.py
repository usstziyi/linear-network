import math
import numpy as np


"""
    计算给定值在特定正态分布下的概率密度 
    实现了标准正态分布的概率密度函数(PDF)

    参数:
    x (float 或 np.ndarray): 输入值，可以是单个数值或 numpy 数组
    mu (float): 正态分布的均值
    sigma (float): 正态分布的标准差
"""

def normal(x, mu, sigma):
    """可用于计算给定值在特定正态分布下的概率密度"""
    # 计算正态分布的归一化常数，即概率密度函数中的分母部分
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    # 计算指数部分并与归一化常数相乘，得到正态分布的概率密度值
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)