import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        # 使用自适应平均池化层将特征图的空间维度压缩到1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 使用一维卷积层对通道维度进行压缩和扩展
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        # 使用sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        # 初始化权重和偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 通过自适应平均池化层
        y = self.gap(x)  # bs,c,1,1
        # 调整形状，准备输入到一维卷积层
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        # 通过一维卷积层
        y = self.conv(y)  # bs,1,c
        # 通过sigmoid激活函数
        y = self.sigmoid(y)  # bs,1,c
        # 调整形状，准备与原始特征图进行元素乘法
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        # 将注意力权重与原始特征图进行元素乘法
        return x * y.expand_as(x)

if __name__ == '__main__':
    # 创建一个随机输入张量，模拟批量大小为50的特征图，每个特征图有512个通道，大小为7x7
    input = torch.randn(50, 512, 7, 7)
    # 创建ECAAttention模型实例
    eca = ECAAttention(kernel_size=3)
    # 通过模型传递输入张量
    output = eca(input)
    # 打印输出的形状
    print(output.shape)
