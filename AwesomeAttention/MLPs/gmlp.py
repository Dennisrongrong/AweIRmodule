""" 
PyTorch implementation of Pay Attention to MLPs

As described in https://arxiv.org/pdf/2105.08050

gMLP, based on MLPs with gating, and show that it can perform as well as Transformers 
in key language and vision applications.
"""



import torch
from torch import nn


# 定义一个名为SpatialGatingUnit的类，它是一个神经网络模块
class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, sequence_len):
        super().__init__()
        # 将输入维度dim除以2得到gate_dim
        gate_dim = dim // 2
        # 实例化一个层归一化对象
        self.norm = nn.LayerNorm(gate_dim)
        # 实例化一个线性变换对象，用于投影
        self.proj = nn.Linear(sequence_len, sequence_len)

    # 初始化权重的方法
    def init_weights(self):
        # 为proj的权重初始化为标准正态分布，标准差为1e-6
        nn.init.normal_(self.proj.weight, std=1e-6)
        # 为proj的偏置初始化为1
        nn.init.ones_(self.proj.bias)

    # 前向传播的方法
    def forward(self, x):
        # 将输入x分割为两个部分u和v，分割维度为-1，即最后一个维度
        u, v = x.chunk(2, dim=-1)
        # 对v进行归一化和线性变换
        v = self.proj(self.norm(v).transpose(-1, -2))
        # 返回u和变换后的v的乘积，注意v需要转置以匹配u的维度
        return u * v.transpose(-1, -2)


# 定义一个名为Block的类，它是一个神经网络模块
class Block(nn.Module):
    def __init__(self, dim, sequence_len, mlp_ratio=4, drop=0):
        super().__init__()
        # 实例化一个层归一化对象
        self.norm = nn.LayerNorm(dim)
        # 计算通道维度，为dim乘以mlp_ratio
        channel_dim = int(dim * mlp_ratio)
        # 实例化两个线性变换对象
        self.fc1 = nn.Linear(dim, channel_dim)
        self.act = nn.GELU()
        # 实例化一个Dropout对象
        self.drop = nn.Dropout(drop, inplace=True)
        # 实例化一个SpatialGatingUnit对象
        self.sgu = SpatialGatingUnit(channel_dim, sequence_len)
        # 实例化另一个线性变换对象
        self.fc2 = nn.Linear(channel_dim // 2, dim)

    # 前向传播的方法
    def forward(self, x):
        # 对输入x进行层归一化
        x = self.norm(x)
        # 通过第一个线性变换和激活函数
        x = self.fc1(x)
        x = self.act(x)
        # 应用Dropout
        x = self.drop(x)
        # 通过SpatialGatingUnit
        x = self.sgu(x)
        # 通过第二个线性变换
        x = self.fc2(x)
        # 再次应用Dropout
        x = self.drop(x)
        # 返回处理后的x
        return x


# 定义一个名为PatchEmbedding的类，它是一个神经网络模块
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=3, embedding_dim=768):
        super().__init__()
        # 确保图像尺寸可以被patch_size整除
        assert image_size % patch_size == 0
        # 计算网格尺寸
        grid_size = image_size // patch_size
        # 计算patch的数量
        self.num_patches = grid_size * grid_size
        # 实例化一个卷积层，用于将输入图像的每个patch映射到embedding_dim维度
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    # 前向传播的方法
    def forward(self, x):
        # 对输入x应用卷积层
        x = self.proj(x)
        # 将x展平并转置，以匹配后续处理的需要
        x = x.flatten(2).transpose(-1, -2)
        # 返回处理后的x
        return x


# 定义一个名为gMLP的类，它是一个神经网络模块
class gMLP(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, dim=768,
                 mlp_ratio=4, drop=0, depths=12, num_classes=1000):
        super().__init__()
        # 实例化PatchEmbedding对象
        self.patch_embedding = PatchEmbedding(image_size, patch_size,
                                              in_channels, dim)
        # 实例化多个Block对象并将它们序列化
        self.blocks = nn.Sequential(*[Block(dim, self.patch_embedding.num_patches,
                                            mlp_ratio, drop) for i in range(depths)])
        # 实例化一个线性变换对象，用于最后的分类
        self.head = nn.Linear(dim, num_classes)

    # 前向传播的方法
    def forward(self, x):
        # 对输入x应用PatchEmbedding
        x = self.patch_embedding(x)
        # 通过多个Block对象
        x = self.blocks(x)
        # 对x在序列长度维度上取平均
        x = x.mean(dim=1)
        # 通过最后的线性变换
        x = self.head(x)
        # 返回处理后的x
        return x


# 如果这个脚本是主程序，则执行以下代码
if __name__ == "__main__":
    # 创建一个随机输入张量x
    x = torch.randn(2, 3, 224, 224)
    # 实例化gMLP模型
    model = gMLP()
    # 通过模型处理输入x
    y = model(x)
    # 打印输出y的形状
    print(y.shape)