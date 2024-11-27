import torch

import torch.nn as nn


class sSE(nn.Module):  # 空间(Space)注意力
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)  # 定义一个卷积层，用于将输入通道转换为单通道
        self.norm = nn.Sigmoid()  # 应用Sigmoid激活函数进行归一化

    def forward(self, x):
        q = self.conv(x)  # 使用卷积层减少通道数至1：b c h w -> b 1 h w
        q = self.norm(q)  # 对卷积后的结果应用Sigmoid激活函数：b 1 h w
        return x * q  # 通过广播机制将注意力权重应用到每个通道上


class cSE(nn.Module):  # 通道(channel)注意力
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 使用自适应平均池化，输出大小为1x1
        self.relu = nn.ReLU()  # ReLU激活函数
        self.Conv_Squeeze = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)  # 通道压缩卷积层
        self.norm = nn.Sigmoid()  # Sigmoid激活函数进行归一化
        self.Conv_Excitation = nn.Conv2d(in_ch // 2, in_ch, kernel_size=1, bias=False)  # 通道激励卷积层

    def forward(self, x):
        z = self.avgpool(x)  # 对输入特征进行全局平均池化：b c 1 1
        z = self.Conv_Squeeze(z)  # 通过通道压缩卷积减少通道数：b c//2 1 1
        z = self.relu(z)  # 应用ReLU激活函数
        z = self.Conv_Excitation(z)  # 通过通道激励卷积恢复通道数：b c 1 1
        z = self.norm(z)  # 对激励结果应用Sigmoid激活函数进行归一化
        return x * z.expand_as(x)  # 将归一化权重乘以原始特征，使用expand_as扩展维度与原始特征相匹配


class scSE(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.cSE = cSE(in_ch)  # 通道注意力模块
        self.sSE = sSE(in_ch)  # 空间注意力模块

    def forward(self, x):
        c_out = self.cSE(x)  # 应用通道注意力
        s_out = self.sSE(x)  # 应用空间注意力
        return c_out + s_out  # 合并通道和空间注意力的输出


x = torch.randn(4, 16, 4, 4)  # 测试输入
net = scSE(16)  # 实例化模型
print(net(x).shape)  # 打印输出形状
