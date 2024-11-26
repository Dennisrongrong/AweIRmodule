import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

class SKAttention(nn.Module):
    """
    Selective Kernel Attention
    """

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        """
        初始化函数，接收通道数channel、卷积核大小列表kernels、降维比例reduction、分组数group和L。
        :param channel: 通道数
        :param kernels: 卷积核大小列表
        :param reduction: 降维比例
        :param group: 分组数
        :param L: 一个较大的数，用于计算d
        """
        super().__init__()
        self.d = max(L, channel // reduction)  # 计算d
        self.convs = nn.ModuleList([])  # 创建卷积层列表
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)  # 创建全连接层
        self.fcs = nn.ModuleList([])  # 创建全连接层列表
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))  # 添加全连接层
        self.softmax = nn.Softmax(dim=0)  # 创建Softmax层

    def forward(self, x):
        """
        前向传播函数，定义Selective Kernel Attention的计算流程。
        :param x: 输入特征图
        :return: 输出特征图
        """
        bs, c, _, _ = x.size()  # 获取输入特征图的尺寸
        conv_outs = []  # 存储卷积输出
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))  # 执行卷积操作
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)  # bs,channel
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel,1,1
        attention_weights = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weights = self.softmax(attention_weights)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weights * feats).sum(0)  # bs,c,h,w
        return V  # 返回输出特征图

if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建一个随机初始化的输入张量
    se = SKAttention(channel=512, reduction=8)  # 创建Selective Kernel Attention模块实例
    output = se(input)  # 通过Selective Kernel Attention模块前向传播
    print(output.shape)  # 打印输出的形状