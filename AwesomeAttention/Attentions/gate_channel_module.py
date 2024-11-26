import torch
import torch.nn.functional as F
import math
from torch import nn

# GCT类定义了Gated Channel Transformation模块
class GCT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        # 初始化可学习的参数alpha、gamma和beta，它们的维度是[1, num_channels, 1, 1]
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon  # 用于避免除以零的小常数
        self.mode = mode  # 归一化模式，可以是'l2'或'l1'
        self.after_relu = after_relu  # 指示GCT是在ReLU激活后应用

    def forward(self, x):
        # 根据模式选择不同的归一化方法
        if self.mode == 'l2':
            # L2归一化：计算每个通道的L2范数，并乘以alpha参数
            embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            # 计算归一化因子
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            # L1归一化：计算每个通道的L1范数，并乘以alpha参数
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            # 计算归一化因子
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        # 计算门控信号
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        # 将门控信号应用于输入特征图
        return x * gate

# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)  # 创建一个随机的输入特征图
    attn = GCT(64)  # 实例化GCT模块，通道数为64
    y = attn(x)  # 通过GCT模块
    print(y.shape)  # 打印输出特征图的形状