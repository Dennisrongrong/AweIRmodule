import numpy as np
import torch
from torch import nn
from torch.nn import init

# 定义AFT_FULL（Attention with Full connections）模块
class AFT_FULL(nn.Module):
    def __init__(self, d_model, n=49, simple=False):
        super(AFT_FULL, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)  # 定义全连接层fc_q，用于计算query
        self.fc_k = nn.Linear(d_model, d_model)  # 定义全连接层fc_k，用于计算key
        self.fc_v = nn.Linear(d_model, d_model)  # 定义全连接层fc_v，用于计算value
        if simple:  # 如果simple为True，使用零作为位置偏置
            self.position_biases = torch.zeros((n, n))
        else:  # 否则，使用可学习的参数作为位置偏置
            self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model  # 模型的维度
        self.n = n  # 序列长度或位置数
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

        self.init_weights()  # 初始化权重

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 应用Kaiming初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.BatchNorm2d):  # 如果是批归一化层
                init.constant_(m.weight, 1)  # 初始化权重为1
                init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                init.normal_(m.weight, std=0.001)  # 应用正态分布初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置为0

    def forward(self, input):
        bs, n, dim = input.shape  # 获取输入张量的形状

        q = self.fc_q(input)  # 通过全连接层计算query
        k = self.fc_k(input).view(1, bs, n, dim)  # 通过全连接层计算key，并调整形状
        v = self.fc_v(input).view(1, bs, n, dim)  # 通过全连接层计算value，并调整形状

        numerator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v, dim=2)  # 计算分子
        denominator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)), dim=2)  # 计算分母

        out = (numerator / denominator)  # 计算输出
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  # 将Sigmoid激活函数应用于query，并与输出相乘

        return out

# 测试代码
if __name__ == '__main__':
    input = torch.randn(50, 49, 512)  # 创建一个随机输入张量（输入为批次，位置索引，特征维度）
    aft_full = AFT_FULL(d_model=512, n=49)  # 实例化AFT_FULL模块
    output = aft_full(input)  # 将输入张量传递给AFT_FULL模块
    print(output.shape)  # 打印输出张量的形状