""" 
PyTorch implementation of Patches Are All You Need?

As described in https://arxiv.org/pdf/2201.09792

ConvMixer, consists of a patch embedding layer followed by repeated applications
of a simple fully-convolutional block.
"""




import torch
from torch import nn

# 定义Block类，表示ConvMixer中的一个基础块，包含深度卷积（depthwise convolution）和点卷积（pointwise convolution）
class Block(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        self.dwconv = nn.Sequential(  # 深度卷积序列
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),  # 深度卷积，保持维度不变
            nn.GELU(),  # GELU激活函数
            nn.BatchNorm2d(dim)  # 批量归一化
        )
        self.pwconv = nn.Sequential(  # 点卷积序列
            nn.Conv2d(dim, dim, 1),  # 点卷积，保持维度不变
            nn.GELU(),  # GELU激活函数
            nn.BatchNorm2d(dim)  # 批量归一化
        )

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)  # 通过深度卷积分支
        x += shortcut  # 添加残差连接
        x = self.pwconv(x)  # 通过点卷积分支
        return x

# 定义ConvMixer类，表示整个ConvMixer模型
class ConvMixer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 in_channels=3,
                 kernel_size=9,
                 patch_size=7,
                 num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(  # 模型的干部分（stem）
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),  # 卷积层，减少空间维度
            nn.GELU(),  # GELU激活函数
            nn.BatchNorm2d(dim)  # 批量归一化
        )
        self.blocks = nn.Sequential(*[Block(dim, kernel_size)  # 多个Block的序列
                                      for i in range(depth)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.fc = nn.Linear(dim, num_classes)  # 全连接层，输出类别数

    def forward(self, x):
        x = self.stem(x)  # 通过模型的干部分
        x = self.blocks(x)  # 通过多个Block
        x = self.avgpool(x)  # 通过自适应平均池化
        x = x.view(x.shape[0], -1)  # 展平特征图
        x = self.fc(x)  # 通过全连接层
        return x

# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)  # 创建一个随机输入张量
    model = ConvMixer(128, 6)  # 创建ConvMixer模型实例，dim=128，depth=6
    y = model(x)  # 模型前向传播
    print(y.shape)  # 打印输出张量的形状