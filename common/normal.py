import math
import numpy as np

"""
normal distribution
正太分布
"""

def normal(x, mean=0, std=1):
    """计算x在均值为mean、标准差为std的正态分布中的概率密度值。

    参数:
        x (float or np.ndarray): 输入值或数组
        mean (float, optional): 正态分布的均值. Defaults to 0.
        std (float, optional): 正态分布的标准差. Defaults to 1.

    返回:
        float or np.ndarray: 输入值或数组在正态分布中的概率密度值
    """
    p = 1 / math.sqrt(2 * math.pi * std**2)
    return p * np.exp(-0.5 / std**2 * (x - mean)**2)
