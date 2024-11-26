# 导入必要的库
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

# 定义一个名为DoubleAttention的类，它继承自PyTorch的nn.Module
class DoubleAttention(nn.Module):
    # 初始化方法，设置类的属性
    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()  # 调用父类的初始化方法
        self.in_channels = in_channels  # 输入通道数
        self.reconstruct = reconstruct  # 是否进行特征重构的标志
        self.c_m = c_m  # 特征门控的中间通道数
        self.c_n = c_n  # 特征分布的中间通道数
        # 定义三个卷积层，分别用于特征门控、特征分布和特征值
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        # 如果需要重构，则添加一个卷积层用于重构特征
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        # 调用初始化权重的方法
        self.init_weights()

    # 初始化权重的方法
    def init_weights(self):
        for m in self.modules():  # 遍历模型中的所有模块
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化权重
                if m.bias is not None:  # 如果有偏置项
                    init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.BatchNorm2d):  # 如果是批量归一化层
                init.constant_(m.weight, 1)  # 初始化权重为1
                init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                init.normal_(m.weight, std=0.001)  # 使用标准正态分布初始化权重
                if m.bias is not None:  # 如果有偏置项
                    init.constant_(m.bias, 0)  # 初始化偏置为0

    # 前向传播方法
    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入的批次大小、通道数、高度和宽度
        assert c == self.in_channels  # 确保输入通道数与定义的一致
        # 通过卷积层获取特征门控、特征分布和特征值
        A = self.convA(x)  # b, c_m, h, w
        B = self.convB(x)  # b, c_n, h, w
        V = self.convV(x)  # b, c_n, h, w
        tmpA = A.view(b, self.c_m, -1)  # 将特征门控的特征展平
        # 使用softmax函数获取注意力图和注意力向量
        attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=2)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=2)
        # 特征门控步骤：通过矩阵乘法获取全局描述符
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # b, c_m, c_n
        # 特征分布步骤：通过矩阵乘法和点积获取分布后的特征
        tmpZ = global_descriptors.matmul(attention_vectors)  # b, c_m, h*w
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # 将分布后的特征恢复到原始空间维度
        # 如果需要重构，则通过卷积层重构特征
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ  # 返回输出特征

# 测试代码
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建一个随机输入张量（输入张量为批次大小，通道数，特征图高度，特征图宽度）
    a2 = DoubleAttention(512, 128, 128, True)  # 实例化DoubleAttention类
    output = a2(input)  # 将输入张量通过模型
    print(output.shape)  # 打印输出张量的形状