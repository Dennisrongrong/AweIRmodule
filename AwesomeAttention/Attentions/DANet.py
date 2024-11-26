import numpy as np
import torch
from torch import nn
from torch.nn import init
from SelfAttention import ScaledDotProductAttention
from SimplifiedSelfAttention import SimplifiedScaledDotProductAttention

# 位置注意力模块
class PositionAttentionModule(nn.Module):
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size-1)//2)  # 卷积层
        self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)  # 缩放点积注意力层

    def forward(self, x):
        bs, c, h, w = x.shape
        y = self.cnn(x)  # 通过卷积层
        y = y.view(bs, c, -1).permute(0, 2, 1)  # 调整形状并转置，准备输入到注意力层
        y = self.pa(y, y, y)  # 通过缩放点积注意力层
        return y

# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size-1)//2)  # 卷积层
        self.pa = SimplifiedScaledDotProductAttention(H*W, h=1)  # 简化的缩放点积注意力层

    def forward(self, x):
        bs, c, h, w = x.shape
        y = self.cnn(x)  # 通过卷积层
        y = y.view(bs, c, -1)  # 调整形状，准备输入到注意力层
        y = self.pa(y, y, y)  # 通过简化的缩放点积注意力层
        return y

# 双重注意力模块
class DAModule(nn.Module):
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.position_attention_module = PositionAttentionModule(d_model=512, kernel_size=3, H=7, W=7)  # 位置注意力模块
        self.channel_attention_module = ChannelAttentionModule(d_model=512, kernel_size=3, H=7, W=7)  # 通道注意力模块

    def forward(self, input):
        bs, c, h, w = input.shape
        p_out = self.position_attention_module(input)  # 通过位置注意力模块
        c_out = self.channel_attention_module(input)  # 通过通道注意力模块
        p_out = p_out.permute(0, 2, 1).view(bs, c, h, w)  # 调整形状
        c_out = c_out.view(bs, c, h, w)  # 调整形状
        return p_out + c_out  # 返回位置和通道注意力的结果之和

if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建一个随机初始化的输入张量
    danet = DAModule(d_model=512, kernel_size=3, H=7, W=7)  # 创建DAModule实例
    print(danet(input).shape)  # 打印输出的形状