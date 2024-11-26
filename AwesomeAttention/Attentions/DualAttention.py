import torch
from torch import nn

# 位置注意力模块（PAM）
class PAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.b = nn.Conv2d(dim, dim, 1)  # 用于生成空间注意力矩阵B的卷积层
        self.c = nn.Conv2d(dim, dim, 1)  # 用于生成特征C的卷积层
        self.d = nn.Conv2d(dim, dim, 1)  # 用于生成特征D的卷积层
        self.alpha = nn.Parameter(torch.zeros(1))  # 用于调整注意力权重的参数

    def forward(self, x):
        n, c, h, w = x.shape  # 获取输入特征图的维度
        B = self.b(x).flatten(2).transpose(1, 2)  # 生成空间注意力矩阵B
        C = self.c(x).flatten(2)  # 生成特征C
        D = self.d(x).flatten(2).transpose(1, 2)  # 生成特征D
        attn = (B @ C).softmax(dim=-1)  # 计算空间注意力权重
        y = (attn @ D).transpose(1, 2).reshape(n, c, h, w)  # 应用注意力权重
        out = self.alpha * y + x  # 将注意力加权的特征与原始特征相加
        return out

# 通道注意力模块（CAM）
class CAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))  # 用于调整注意力权重的参数

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入特征图的维度
        x_ = x.flatten(2)  # 将特征图展平
        attn = torch.matmul(x_, x_.transpose(1, 2))  # 计算通道注意力权重
        attn = attn.softmax(dim=-1)  # 应用softmax获取注意力权重
        x_ = (attn @ x_).reshape(b, c, h, w)  # 应用注意力权重
        out = self.beta * x_ + x  # 将注意力加权的特征与原始特征相加
        return out

# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)  # 创建一个随机的输入特征图
    #attn = PAM(64)  # 实例化PAM模块
    attn = CAM()  # 实例化CAM模块
    y = attn(x)  # 通过注意力模块
    print(y.shape)  # 打印输出特征图的形状