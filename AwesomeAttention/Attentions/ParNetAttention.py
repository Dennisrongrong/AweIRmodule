import numpy as np
import torch
from torch import nn
from torch.nn import init

class ParNetAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        # 自适应平均池化后接1x1卷积和Sigmoid激活函数，用于生成注意力图
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化
            nn.Conv2d(channel, channel, kernel_size=1),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 1x1卷积后接批量归一化，用于特征变换
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),  # 1x1卷积
            nn.BatchNorm2d(channel)  # 批量归一化
        )

        # 3x3卷积后接批量归一化，用于特征变换
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),  # 3x3卷积，padding为1
            nn.BatchNorm2d(channel)  # 批量归一化
        )

        self.silu = nn.SiLU()  # SiLU激活函数

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入张量的尺寸
        x1 = self.conv1x1(x)  # 通过1x1卷积
        x2 = self.conv3x3(x)  # 通过3x3卷积
        x3 = self.sse(x) * x  # 通过自适应平均池化和1x1卷积生成注意力图，与原特征相乘
        y = self.silu(x1 + x2 + x3)  # 将三个特征图相加，并通过SiLU激活函数
        return y

if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建一个随机输入张量
    pna = ParNetAttention(channel=512)  # 实例化ParNetAttention模块
    output = pna(input)  # 通过模型传递输入，获取输出
    print(output.shape)  # 打印输出张量的形状