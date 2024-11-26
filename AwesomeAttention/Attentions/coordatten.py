import torch
from torch import nn

# 坐标注意力类
class CoordinateAttention(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 垂直方向的自适应平均池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 水平方向的自适应平均池化
        hidden_dim = max(8, in_dim // reduction)  # 计算隐藏层维度
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(hidden_dim)  # 批归一化层
        self.act = nn.ReLU(inplace=True)  # 激活函数
        self.conv_h = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)  # 垂直方向的卷积层
        self.conv_w = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)  # 水平方向的卷积层

    def forward(self, x):
        identity = x  # 保存输入x，用于后面的残差连接
        b, c, h, w = x.shape  # 获取输入x的维度
        x_h = self.pool_h(x)  # 对x进行垂直方向的池化
        x_w = self.pool_w(x).transpose(-1, -2)  # 对x进行水平方向的池化并转置
        y = torch.cat([x_h, x_w], dim=2)  # 将垂直和水平方向的池化结果拼接
        y = self.conv1(y)  # 通过第一个卷积层
        y = self.bn1(y)  # 通过批归一化层
        y = self.act(y)  # 通过激活函数
        x_h, x_w = torch.split(y, [h, w], dim=2)  # 将拼接的结果分割为垂直和水平两部分
        x_w = x_w.transpose(-1, -2)  # 将水平部分转置回原来的形状
        a_h = self.conv_h(x_h)  # 通过垂直方向的卷积层
        a_w = self.conv_w(x_w)  # 通过水平方向的卷积层
        out = identity * a_h * a_w  # 将输入x与两个方向的注意力权重相乘，得到最终输出
        return out

# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)  # 创建一个随机的输入特征图
    attn = CoordinateAttention(64, 64)  # 实例化坐标注意力模块，输入和输出通道数为64
    y = attn(x)  # 通过坐标注意力模块
    print(y.shape)  # 打印输出特征图的形状