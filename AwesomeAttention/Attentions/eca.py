import torch
from torch import nn
import math

# ECALayer类定义了ECA-Net中的高效通道注意力（ECA）层
class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层，用于全局平均池化
        t = int(abs((math.log(channels, 2) + b) / gamma))  # 计算1D卷积的核大小
        k = t if t % 2 else t + 1  # 确保核大小为奇数
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)  # 1D卷积层
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        y = self.avgpool(x)  # 对输入特征图进行全局平均池化
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 通过1D卷积层
        y = self.sigmoid(y)  # 应用Sigmoid激活函数
        return x * y.expand_as(x)  # 将注意力权重应用于输入特征图

# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)  # 创建一个随机的输入特征图
    attn = ECALayer(64)  # 实例化ECA层，通道数为64
    y = attn(x)  # 通过ECA层
    print(y.shape)  # 打印输出特征图的形状