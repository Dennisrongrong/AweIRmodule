import torch
from torch import nn
import torch.nn.functional as F

# ChannelGate 类定义了通道注意力机制
class ChannelGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，用于全局信息的聚合
        self.mlp = nn.Sequential(  # 多层感知机，用于学习通道间的关系
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        self.bn = nn.BatchNorm1d(channel)  # 批归一化层，用于通道注意力的输出

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入特征图的维度
        y = self.avgpool(x).view(b, c)  # 将特征图通过自适应平均池化后展平
        y = self.mlp(y)  # 通过多层感知机处理
        y = self.bn(y).view(b, c, 1, 1)  # 通过批归一化并重塑形状
        return y.expand_as(x)  # 扩展到与输入特征图相同的形状

# SpatialGate 类定义了空间注意力机制
class SpatialGate(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3, dilation_val=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)  # 1x1卷积用于降维
        self.conv2 = nn.Sequential(  # 空间注意力的卷积层
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(channel // reduction, 1, kernel_size=1)  # 最后一个1x1卷积，输出空间注意力图
        self.bn = nn.BatchNorm2d(1)  # 批归一化层，用于空间注意力的输出

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入特征图的维度
        y = self.conv1(x)  # 通过1x1卷积降维
        y = self.conv2(y)  # 通过空间注意力的卷积层
        y = self.conv3(y)  # 通过最后的1x1卷积
        y = self.bn(y)  # 通过批归一化
        return y.expand_as(x)  # 扩展到与输入特征图相同的形状

# BAM 类定义了Bottleneck Attention Module
class BAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_attn = ChannelGate(channel)  # 实例化通道注意力
        self.spatial_attn = SpatialGate(channel)  # 实例化空间注意力

    def forward(self, x):
        attn = F.sigmoid(self.channel_attn(x) + self.spatial_attn(x))  # 计算最终的注意力图
        return x + x * attn  # 将注意力图应用于输入特征图

# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)  # 创建一个随机的输入特征图
    attn = BAM(64)  # 实例化BAM模块，通道数为64
    y = attn(x)  # 通过BAM模块
    print(y.shape)  # 打印输出特征图的形状
