import numpy as np
import torch
from torch import nn
from torch.nn import init

# 定义ResidualAttention类，它继承自PyTorch的nn.Module
class ResidualAttention(nn.Module):
    def __init__(self, channel=512, num_class=1000, la=0.2):
        super().__init__()
        self.la = la  # la是用于调整平均值和最大值之间权重的超参数
        self.fc = nn.Conv2d(in_channels=channel, out_channels=num_class, kernel_size=1, stride=1, bias=False)  # 定义一个全连接层，实际上是一个卷积层，用于分类

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入x的维度
        y_raw = self.fc(x).flatten(2)  # 通过卷积层后，将输出扁平化，维度变为[b, num_class, h*w]
        y_avg = torch.mean(y_raw, dim=2)  # 计算每个类别的平均值，维度变为[b, num_class]
        y_max = torch.max(y_raw, dim=2)[0]  # 计算每个类别的最大值，维度变为[b, num_class]
        score = y_avg + self.la * y_max  # 将平均值和最大值结合，形成最终的分数
        return score

# 测试代码
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建一个随机输入张量
    resatt = ResidualAttention(channel=512, num_class=1000, la=0.2)  # 创建ResidualAttention模块实例
    output = resatt(input)  # 将输入传递给ResidualAttention模块
    print(output.shape)  # 打印输出张量的形状