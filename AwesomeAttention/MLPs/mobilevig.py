""" 
PyTorch implementation of MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications

As described in https://arxiv.org/pdf/2307.00395.pdf

The authors propose a novel mobile CNN-GNN architecture
for vision tasks using our proposed SVGA, maxrelative graph convolution, 
and concepts from mobile CNN and mobile vision transformer architectures.
"""





import torch
from torch import nn


# 定义一个MBConvBlock类，表示MobileNet中的深度可分离卷积块
class MBConvBlock(nn.Module):
    def __init__(self, dim, expand_ratio=4):
        super().__init__()
        # 1x1卷积，用于扩展通道数
        self.conv1 = nn.Conv2d(dim, dim * expand_ratio, 1)
        self.bn1 = nn.BatchNorm2d(dim * expand_ratio)  # 批归一化
        # 3x3深度卷积，使用分组卷积
        self.conv2 = nn.Conv2d(dim * expand_ratio, dim * expand_ratio, kernel_size=3,
                               stride=1, padding=1, groups=dim * expand_ratio)
        self.bn2 = nn.BatchNorm2d(dim * expand_ratio)  # 批归一化
        # 1x1卷积，用于恢复通道数
        self.conv3 = nn.Conv2d(dim * expand_ratio, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)  # 批归一化
        self.act = nn.GELU()  # 激活函数

    def forward(self, x):
        shortcut = x  # 保存输入以便后续残差连接
        # 执行卷积、激活和归一化
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) + shortcut  # 残差连接
        return x

    # 定义一个MRConv4d类，表示Max-Relative图卷积


class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751)  for dense data type

    K是超补丁的数量，因此hops等于res // K。
    """

    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        # 定义一个序列，包括1x1卷积、批归一化和激活函数
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )
        self.K = K  # 超补丁数量

    def forward(self, x):
        B, C, H, W = x.shape  # 获取输入的形状

        x_j = x - x  # 初始化x_j为零张量
        # 在高度上进行Max-Relative卷积
        for i in range(self.K, H, self.K):
            x_c = x - torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)  # 计算相对特征
            x_j = torch.max(x_j, x_c)  # 更新x_j为最大值
        # 在宽度上进行Max-Relative卷积
        for i in range(self.K, W, self.K):
            x_r = x - torch.cat([x[:, :, :, -i:], x[:, :, :, :-i]], dim=3)  # 计算相对特征
            x_j = torch.max(x_j, x_r)  # 更新x_j为最大值

        x = torch.cat([x, x_j], dim=1)  # 将原始特征和相对特征拼接
        return self.nn(x)  # 通过定义的神经网络层


# 定义一个Grapher类，表示图卷积模块
class Grapher(nn.Module):
    def __init__(self, dim, k=2):
        super().__init__()
        # 定义输入卷积层和批归一化
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
        )
        # 定义Max-Relative图卷积层
        self.graph_conv = MRConv4d(dim, dim * 2, k)
        # 定义输出卷积层和批归一化
        self.fc2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        shortcut = x  # 保存输入以便后续残差连接
        x = self.fc1(x)  # 通过第一个卷积层
        x = self.graph_conv(x)  # 通过图卷积层
        x = self.fc2(x)  # 通过第二个卷积层
        x += shortcut  # 残差连接
        return x


# 定义一个FFN类，表示前馈神经网络
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features  # 如果未指定，则使用输入特征数
        out_features = out_features or in_features  # 如果未指定，则使用输入特征数
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 1x1卷积
        self.act = nn.GELU()  # 激活函数
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)  # 1x1卷积

    def forward(self, x):
        x = self.fc1(x)  # 通过第一个卷积层
        x = self.act(x)  # 激活
        x = self.fc2(x)  # 通过第二个卷积层
        return x

# 定义SVGABlock类，这是一个结合了图卷积和前馈网络的模块。
class SVGABlock(nn.Module):
    def __init__(self, dim, k=2):
        super().__init__()
        self.grapher = Grapher(dim, k)  # 初始化图卷积模块Grapher
        self.ffn = FFN(dim, dim * 4)  # 初始化前馈网络FFN，隐藏层维度是输入维度的4倍

    def forward(self, x):
        x = self.grapher(x)  # 将输入x通过图卷积模块
        x = self.ffn(x)  # 再将图卷积的输出通过前馈网络
        return x  # 返回前馈网络的输出


# 定义MobileViG类，这是一个基于MobileNet和图卷积的混合模型。
class MobileViG(nn.Module):
    def __init__(self, embed_dims=[42, 84, 168, 256], depths=[2, 2, 6, 2],
                 k=2, num_classes=1000):
        super().__init__()
        self.downsamples = nn.ModuleList()  # 用于存储下采样模块的列表
        # 定义模型的Stem部分，即初始的下采样层
        stem = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )
        self.downsamples.append(stem)  # 将Stem部分添加到下采样模块列表
        # 定义后续的下采样层
        for i in range(3):
            downsample = nn.Sequential(
                nn.Conv2d(embed_dims[i], embed_dims[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dims[i + 1])
            )
            self.downsamples.append(downsample)

        self.stages = nn.ModuleList()  # 用于存储不同阶段的模块列表
        # 根据给定的深度和嵌入维度，定义每个阶段的模块
        for i in range(4):
            if i == 3:
                # 最后一个阶段使用SVGABlock
                layer = nn.Sequential(*[SVGABlock(embed_dims[i], k) for j in range(depths[i])])
                self.stages.append(layer)
            else:
                # 其他阶段使用MBConvBlock
                layer = nn.Sequential(*[MBConvBlock(embed_dims[i]) for j in range(depths[i])])
                self.stages.append(layer)
        self.head = nn.Linear(embed_dims[-1], num_classes)  # 定义分类头，将最后一个阶段的输出映射到类别数

    def forward(self, x):
        # 通过下采样层和各个阶段的模块进行前向传播
        for i in range(4):
            x = self.downsamples[i](x)  # 通过下采样层
            x = self.stages[i](x)  # 通过对应阶段的模块
        x = x.mean(dim=[-1, -2])  # 在空间维度上取平均，减少维度
        x = self.head(x)  # 通过分类头，得到最终的分类结果
        return x  # 返回分类结果

import torch
from torch import nn

# 定义MBConvBlock、MRConv4d、Grapher、FFN、SVGABlock和MobileViG类（代码省略）

# 定义一个函数，返回一个MobileViG模型实例，配置为ti版本
def mobilevig_ti(num_classes=1000):
    return MobileViG(embed_dims=[42, 84, 168, 256], depths=[2, 2, 6, 2], k=2, num_classes=num_classes)

# 定义一个函数，返回一个MobileViG模型实例，配置为s版本
def mobilevig_s(num_classes=1000):
    return MobileViG(embed_dims=[42, 84, 176, 256], depths=[3, 3, 9, 3], k=2, num_classes=num_classes)

# 定义一个函数，返回一个MobileViG模型实例，配置为m版本
def mobilevig_m(num_classes=1000):
    return MobileViG(embed_dims=[42, 84, 224, 400], depths=[3, 3, 9, 3], k=2, num_classes=num_classes)

# 定义一个函数，返回一个MobileViG模型实例，配置为b版本
def mobilevig_b(num_classes=1000):
    return MobileViG(embed_dims=[42, 84, 240, 464], depths=[5, 5, 15, 5], k=2, num_classes=num_classes)

# 判断是否为主程序运行
if __name__ == "__main__":
    # 创建一个随机数据张量，模拟输入数据
    x = torch.randn(2, 3, 224, 224)
    # 使用mobilevig_ti函数创建一个ti配置的MobileViG模型实例
    model = mobilevig_ti(num_classes=1000)
    # 将随机数据张量通过模型进行前向传播
    y = model(x)
    # 打印输出y的形状
    print(y.shape)