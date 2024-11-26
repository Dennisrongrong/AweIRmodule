import numpy as np
import torch
from torch import nn
from torch.nn import init

# 定义一个空间位移函数spatial_shift1
def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c//4] = x[:, :w-1, :, :c//4]
    x[:, :w-1, :, c//4:c//2] = x[:, 1:, :, c//4:c//2]
    x[:, :, 1:, c//2:c*3//4] = x[:, :, :h-1, c//2:c*3//4]
    x[:, :, :h-1, 3*c//4:] = x[:, :, 1:, 3*c//4:]
    return x

# 定义另一个空间位移函数spatial_shift2
def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c//4] = x[:, :, :h-1, :c//4]
    x[:, :, :h-1, c//4:c//2] = x[:, :, 1:, c//4:c//2]
    x[:, 1:, :, c//2:c*3//4] = x[:, :w-1, :, c//2:c*3//4]
    x[:, :w-1, :, 3*c//4:] = x[:, 1:, :, 3*c//4:]
    return x

# 定义SplitAttention类
class SplitAttention(nn.Module):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)  # 第一个线性层
        self.gelu = nn.GELU()  # GELU激活函数
        self.mlp2 = nn.Linear(channel, channel*k, bias=False)  # 第二个线性层
        self.softmax = nn.Softmax(1)  # Softmax层

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # 重塑输入
        a = torch.sum(torch.sum(x_all, 1), 1)  # 计算所有通道的和
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # 通过两个线性层和GELU激活函数
        hat_a = hat_a.reshape(b, self.k, c)  # 重塑输出
        bar_a = self.softmax(hat_a)  # 应用Softmax
        attention = bar_a.unsqueeze(-2)  # 添加一个维度
        out = attention * x_all  # 计算注意力加权的输入
        out = torch.sum(out, 1).reshape(b, h, w, c)  # 求和并重塑输出
        return out

# 定义S2Attention类
class S2Attention(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels*3)  # 第一个线性层
        self.mlp2 = nn.Linear(channels, channels)  # 第二个线性层
        self.split_attention = SplitAttention()  # SplitAttention模块

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)  # 改变维度顺序
        x = self.mlp1(x)  # 通过第一个线性层
        x1 = spatial_shift1(x[:, :, :, :c])  # 应用空间位移1
        x2 = spatial_shift2(x[:, :, :, c:c*2])  # 应用空间位移2
        x3 = x[:, :, :, c*2:]  # 原始输入的第三部分
        x_all = torch.stack([x1, x2, x3], 1)  # 堆叠三个部分
        a = self.split_attention(x_all)  # 通过SplitAttention模块
        x = self.mlp2(a)  # 通过第二个线性层
        x = x.permute(0, 3, 1, 2)  # 恢复维度顺序
        return x

# 测试代码
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建一个随机输入张量
    s2att = S2Attention(channels=512)  # 创建S2Attention模块实例
    output = s2att(input)  # 将输入传递给S2Attention模块
    print(output.shape)  # 打印输出张量的形状