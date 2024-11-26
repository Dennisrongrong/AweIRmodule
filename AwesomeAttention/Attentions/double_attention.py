import torch
from torch import nn
import torch.nn.functional as F

# DoubleAttention类定义了双注意力网络
class DoubleAttention(nn.Module):
    
    def __init__(self, in_channels, c_m, c_n):
        super().__init__()
        self.c_m = c_m  # 中间特征通道数c_m
        self.c_n = c_n  # 中间特征通道数c_n
        self.in_channels = in_channels  # 输入通道数
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size=1)  # 用于生成特征A的卷积层
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size=1)  # 用于生成特征B的卷积层
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size=1)  # 用于生成特征V的卷积层
        self.proj = nn.Conv2d(c_m, in_channels, kernel_size=1)  # 用于将输出特征投影回输入空间的卷积层

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入特征图的维度
        A = self.convA(x)  # 通过卷积层生成特征A
        B = self.convB(x)  # 通过卷积层生成特征B
        V = self.convV(x)  # 通过卷积层生成特征V
        tmpA = A.view(b, self.c_m, h * w)  # 将特征A重塑为(b, c_m, h * w)
        attention_maps = B.view(b, self.c_n, h * w)  # 将特征B重塑为(b, c_n, h * w)，用于生成注意力图
        attention_vectors = V.view(b, self.c_n, h * w)  # 将特征V重塑为(b, c_n, h * w)，用于生成注意力向量
        attention_maps = F.softmax(attention_maps, dim=-1)  # 在最后一个维度上应用softmax，生成注意力图
        # 第一步：特征聚集
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # 通过矩阵乘法聚集全局特征
        # 第二步：特征分布
        attention_vectors = F.softmax(attention_vectors, dim=1)  # 在c_n维度上应用softmax，生成注意力向量
        tmpZ = global_descriptors.matmul(attention_vectors)  # 通过矩阵乘法分布特征
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # 将分布的特征重塑回(b, c_m, h, w)
        tmpZ = self.proj(tmpZ)  # 将输出特征投影回输入空间
        return tmpZ

# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)  # 创建一个随机的输入特征图
    attn = DoubleAttention(64, 32, 32)  # 实例化双注意力模块，输入通道数为64，c_m和c_n都为32
    y = attn(x)  # 通过双注意力模块
    print(y.shape)  # 打印输出特征图的形状