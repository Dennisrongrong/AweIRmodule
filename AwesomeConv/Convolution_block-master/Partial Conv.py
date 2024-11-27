import os
import sys
import inspect

from torch import nn
import torch

#论文《Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks》

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):

        super().__init__()
        self.dim_conv3 = dim // n_div  # 计算要应用3x3卷积操作的通道数，即输入通道数除以n_div
        self.dim_untouched = dim - self.dim_conv3  # 剩余不进行卷积操作的通道数
        # 初始化一个3x3的卷积层，该卷积层仅作用于部分通道
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        # 根据前向传播策略（通过`forward`参数指定）选择相应的前向传播方法
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError  # 如果提供了未知的前向传播策略，则抛出异常

    def forward_slicing(self, x):
        # 使用切片的方式进行前向传播
        x = x.clone()  # 克隆输入张量，确保不会修改原始输入
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])  # 只对输入张量的一部分通道应用卷积操作
        return x

    def forward_split_cat(self, x):
        # 使用分割和拼接的方式进行前向传播
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)  # 将输入特征图分为两部分
        x1 = self.partial_conv3(x1)  # 对第一部分应用卷积
        x = torch.cat((x1, x2), 1)  # 将处理后的第一部分和未处理的第二部分拼接
        return x


if __name__ == '__main__':
    block = Partial_conv3(64, 2, 'split_cat').cuda()  # 实例化Partial_conv3模块，指定分割和拼接的前向传播策略
    input = torch.rand(1, 64, 64, 64).cuda()  # 创建一个随机的输入张量
    output = block(input)
    print(output.shape)#torch.Size([1, 64, 64, 64])
