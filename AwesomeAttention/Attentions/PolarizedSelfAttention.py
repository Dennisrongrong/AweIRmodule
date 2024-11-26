# 导入必要的库
import numpy as np
import torch
from torch import nn
from torch.nn import init

# 定义一个并行极化自注意力（Parallel Polarized Self-Attention）模块
class ParallelPolarizedSelfAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        # 定义通道注意力的权重和查询的卷积层
        self.ch_wv = nn.Conv2d(channel, channel//2, kernel_size=(1,1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1,1))
        # 定义通道注意力的softmax层
        self.softmax_channel = nn.Softmax(1)
        # 定义空间注意力的softmax层
        self.softmax_spatial = nn.Softmax(-1)
        # 定义通道注意力的输出卷积层
        self.ch_wz = nn.Conv2d(channel//2, channel, kernel_size=(1,1))
        # 定义层归一化
        self.ln = nn.LayerNorm(channel)
        # 定义sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        # 定义空间注意力的权重和查询的卷积层
        self.sp_wv = nn.Conv2d(channel, channel//2, kernel_size=(1,1))
        self.sp_wq = nn.Conv2d(channel, channel//2, kernel_size=(1,1))
        # 定义自适应平均池化层
        self.agp = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        # 通道注意力部分
        channel_wv = self.ch_wv(x)  # bs, c//2, h, w
        channel_wq = self.ch_wq(x)  # bs, 1, h, w
        channel_wv = channel_wv.reshape(b, c//2, -1)  # bs, c//2, h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs, h*w, 1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs, c//2, 1, 1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1)  # bs, c, 1, 1
        channel_out = channel_weight * x

        # 空间注意力部分
        spatial_wv = self.sp_wv(x)  # bs, c//2, h, w
        spatial_wq = self.sp_wq(x)  # bs, c//2, h, w
        spatial_wq = self.agp(spatial_wq)  # bs, c//2, 1, 1
        spatial_wv = spatial_wv.reshape(b, c//2, -1)  # bs, c//2, h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c//2)  # bs, 1, c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs, 1, h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs, 1, h, w
        spatial_out = spatial_weight * x
        out = spatial_out + channel_out
        return out

# 定义一个顺序极化自注意力（Sequential Polarized Self-Attention）模块
class SequentialPolarizedSelfAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        # 定义通道注意力的权重和查询的卷积层
        self.ch_wv = nn.Conv2d(channel, channel//2, kernel_size=(1,1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1,1))
        # 定义通道注意力的softmax层
        self.softmax_channel = nn.Softmax(1)
        # 定义空间注意力的softmax层
        self.softmax_spatial = nn.Softmax(-1)
        # 定义通道注意力的输出卷积层
        self.ch_wz = nn.Conv2d(channel//2, channel, kernel_size=(1,1))
        # 定义层归一化
        self.ln = nn.LayerNorm(channel)
        # 定义sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        # 定义空间注意力的权重和查询的卷积层
        self.sp_wv = nn.Conv2d(channel, channel//2, kernel_size=(1,1))
        self.sp_wq = nn.Conv2d(channel, channel//2, kernel_size=(1,1))
        # 定义自适应平均池化层
        self.agp = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        # 通道注意力部分
        channel_wv = self.ch_wv(x)  # bs, c//2, h, w
        channel_wq = self.ch_wq(x)  # bs, 1, h, w
        channel_wv = channel_wv.reshape(b, c//2, -1)  # bs, c//2, h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs, h*w, 1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs, c//2, 1, 1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1)  # bs, c, 1, 1
        channel_out = channel_weight * x

        # 空间注意力部分，注意这里使用的是channel_out作为输入
        spatial_wv = self.sp_wv(channel_out)  # bs, c//2, h, w
        spatial_wq = self.sp_wq(channel_out)  # bs, c//2, h, w
        spatial_wq = self.agp(spatial_wq)  # bs, c//2, 1, 1
        spatial_wv = spatial_wv.reshape(b, c//2, -1)  # bs, c//2, h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c//2)  # bs, 1, c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs, 1, h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs, 1, h, w
        spatial_out = spatial_weight * channel_out
        return spatial_out

# 测试代码
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    psa = SequentialPolarizedSelfAttention(channel=512)
    output = psa(input)
    print(output.shape)