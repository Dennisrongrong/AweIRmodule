import torch
import torch.nn as nn
import torch.nn.functional as F


# h_sigmoid 类定义了一个硬sigmoid激活函数，其输出范围在[0, 1]
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


# h_swish 类定义了一个基于h_sigmoid的h-swish激活函数
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# CoordAtt 类定义了坐标注意力模块
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 垂直方向的自适应平均池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 水平方向的自适应平均池化

        mip = max(8, inp // reduction)  # 计算中间维度

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(mip)  # 批归一化层
        self.act = h_swish()  # h-swish激活函数

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 垂直方向的卷积层
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 水平方向的卷积层

    def forward(self, x):
        identity = x  # 保存输入x，用于后面的残差连接

        n, c, h, w = x.size()  # 获取输入x的维度
        x_h = self.pool_h(x)  # 对x进行垂直方向的池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 对x进行水平方向的池化并转置