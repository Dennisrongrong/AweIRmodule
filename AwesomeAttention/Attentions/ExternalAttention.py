import numpy as np
import torch
from torch import nn
from torch.nn import init

# 定义一个名为 ExternalAttention 的类，它继承自 nn.Module，是一个神经网络模块。
class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        # 初始化一个线性层 mk，用于将输入的特征维度 d_model 映射到 S 维的注意力空间。
        self.mk = nn.Linear(d_model, S, bias=False)
        # 初始化一个线性层 mv，用于将注意力空间的特征映射回原始的特征维度 d_model。
        self.mv = nn.Linear(S, d_model, bias=False)
        # 初始化 Softmax 激活函数，用于计算注意力权重。
        self.softmax = nn.Softmax(dim=1)
        # 调用初始化权重的方法。
        self.init_weights()

    def init_weights(self):
        # 遍历模块中的所有子模块。
        for m in self.modules():
            # 如果是卷积层，则使用 Kaiming 初始化方法初始化权重。
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # 如果是批量归一化层，则将权重初始化为 1，偏置初始化为 0。
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            # 如果是线性层，则使用标准正态分布初始化权重，并将偏置初始化为 0。
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        # 前向传播函数，接收输入 queries。
        attn = self.mk(queries)  # bs, n, S
        # 应用 Softmax 函数计算注意力权重。
        attn = self.softmax(attn)  # bs, n, S
        # 对注意力权重进行归一化，使得每个样本的权重和为 1。
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs, n, S
        # 将归一化的注意力权重通过线性层 mv 映射回原始特征维度。
        out = self.mv(attn)  # bs, n, d_model
        return out

if __name__ == '__main__':
    # 创建一个随机输入张量，形状为 (batch_size, sequence_length, feature_dim)。
    input = torch.randn(50, 49, 512)
    # 实例化 ExternalAttention 模块，传入特征维度 d_model 和 S。
    ea = ExternalAttention(d_model=512, S=8)
    # 通过模型传递输入，获取输出。
    output = ea(input)
    # 打印输出张量的形状。
    print(output.shape)