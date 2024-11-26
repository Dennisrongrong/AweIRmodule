import numpy as np  # 导入numpy库，用于数学运算
import torch  # 导入PyTorch库，用于深度学习模型的构建
from torch import nn  # 从torch库中导入nn模块，包含神经网络构建所需的类和函数
from torch.nn import init  # 从torch.nn模块中导入init，用于权重初始化


class SpatialGroupEnhance(nn.Module):
    """
    空间分组增强模块，用于增强特征图的空间信息。
    """

    def __init__(self, groups):
        """
        初始化函数，接收分组数groups。
        :param groups: 分组的数量
        """
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层，用于计算每个分组的平均值
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))  # 创建可学习的权重参数
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))  # 创建可学习的偏置参数
        self.sig = nn.Sigmoid()  # 创建Sigmoid激活函数
        self.init_weights()  # 初始化权重

    def init_weights(self):
        """
        自定义权重初始化函数。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.BatchNorm2d):  # 如果是批量归一化层
                init.constant_(m.weight, 1)  # 初始化权重为1
                init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                init.normal_(m.weight, std=0.001)  # 初始化权重为正态分布
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置为0

    def forward(self, x):
        """
        前向传播函数，定义空间分组增强的计算流程。
        :param x: 输入特征图 (b, c, h, w)
        :return: 增强后的特征图
        """
        b, c, h, w = x.shape  # 获取输入特征图的尺寸
        x = x.view(b * self.groups, -1, h, w)  # 调整特征图的形状 (bs*g, dim//g, h, w)
        xn = x * self.avg_pool(x)  # 计算每个分组的平均值并相乘 (bs*g, dim//g, h, w)
        xn = xn.sum(dim=1, keepdim=True)  # 计算每个分组的总和 (bs*g, 1, h, w)
        t = xn.view(b * self.groups, -1)  # 调整形状 (bs*g, h*w)

        t = t - t.mean(dim=1, keepdim=True)  # 减去均值 (bs*g, h*w)
        std = t.std(dim=1, keepdim=True) + 1e-5  # 计算标准差并加上一个小值防止除以0
        t = t / std  # 标准化 (bs*g, h*w)
        t = t.view(b, self.groups, h, w)  # 调整形状 (bs, g, h*w)

        t = t * self.weight + self.bias  # 应用可学习的权重和偏置 (bs, g, h*w)
        t = t.view(b * self.groups, 1, h, w)  # 调整形状 (bs*g, 1, h, w)
        x = x * self.sig(t)  # 应用Sigmoid激活函数并相乘
        x = x.view(b, c, h, w)  # 调整形状回到原始形状 (b, c, h, w)

        return x  # 返回增强后的特征图


if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建一个随机初始化的输入张量
    sge = SpatialGroupEnhance(groups=8)  # 创建空间分组增强模块实例
    output = sge(input)  # 通过空间分组增强模块前向传播
    print(output.shape)  # 打印输出的形状