import torch
from torch import nn

# 定义GCT类，继承自PyTorch的nn.Module
class GCT(nn.Module):
    def __init__(self, channels, c=2, eps=1e-5):
        super().__init__()
        # 使用自适应平均池化层，将特征图的每个通道的空间维度（高和宽）缩减到1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 设置一个很小的常数eps，用于数值稳定性，防止除以0
        self.eps = eps
        # 设置高斯函数的标准差c，控制通道间的注意力激活差异
        self.c = c

    def forward(self, x):
        # 通过自适应平均池化层，对输入特征图x进行全局平均池化
        y = self.avgpool(x)
        # 计算池化后特征的均值
        mean = y.mean(dim=1, keepdim=True)
        # 计算池化后特征的平方的均值
        mean_x2 = (y ** 2).mean(dim=1, keepdim=True)
        # 计算方差
        var = mean_x2 - mean ** 2
        # 归一化池化后的特征，使其具有均值为0，方差为1
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        # 使用高斯函数对归一化的特征进行变换，得到注意力激活
        y_transform = torch.exp(-(y_norm ** 2 / 2 * self.c))
        # 将注意力激活与原始特征图x相乘，实现特征增强
        return x * y_transform.expand_as(x)

# 测试代码
if __name__ == "__main__":
    # 创建一个随机的输入张量x，模拟一个batch size为2，64通道，32x32的特征图
    x = torch.randn(2, 64, 32, 32)
    # 实例化GCT模块，传入通道数为64
    attn = GCT(64)
    # 通过GCT模块处理输入x
    y = attn(x)
    # 打印输出y的形状，应该与输入x的形状相同
    print(y.shape)