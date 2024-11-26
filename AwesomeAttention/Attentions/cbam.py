import torch
from torch import nn

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，用于全局信息聚合
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化，用于突出最显著特征
        self.fc = nn.Sequential(  # 通过两个卷积层实现的特征变换
            nn.Conv2d(channel, channel // reduction, 1, bias=False),  # 降维
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(channel // reduction, channel, 1, bias=False)  # 恢复维度
        )
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，用于输出注意力权重

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 对平均池化的结果进行特征变换
        max_out = self.fc(self.max_pool(x))  # 对最大池化的结果进行特征变换
        out = avg_out + max_out  # 将两个结果相加
        return x * self.sigmoid(out)  # 将注意力权重应用于输入特征图

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)  # 卷积层，用于空间注意力
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道维度上计算平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道维度上计算最大值
        out = torch.concat([avg_out, max_out], dim=1)  # 将平均值和最大值在通道维度上拼接
        out = self.conv(out)  # 通过卷积层计算空间注意力
        return x * self.sigmoid(out)  # 将空间注意力权重应用于输入特征图

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)  # 实例化通道注意力模块
        self.sa = SpatialAttention(kernel_size)  # 实例化空间注意力模块
    
    def forward(self, x):
        x = self.ca(x)  # 应用通道注意力
        x = self.sa(x)  # 应用空间注意力
        return x  # 返回注意力加权后的特征图

# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)  # 创建一个随机的输入特征图
    attn = CBAM(64)  # 实例化CBAM模块，通道数为64
    y = attn(x)  # 通过CBAM模块
    print(y.shape)  # 打印输出特征图的形状