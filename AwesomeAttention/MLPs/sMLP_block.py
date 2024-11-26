import torch
from torch import nn

import torch
from torch import nn


# 定义 sMLPBlock 类
class sMLPBlock(nn.Module):
    def __init__(self, h=224, w=224, c=3):
        super().__init__()
        # 初始化水平方向的线性投影层
        self.proj_h = nn.Linear(h, h)
        # 初始化垂直方向的线性投影层
        self.proj_w = nn.Linear(w, w)
        # 初始化融合层，将三个输入融合在一起
        self.fuse = nn.Linear(3 * c, c)

    def forward(self, x):
        # 对输入 x 的高度维度进行线性变换
        x_h = self.proj_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # 对输入 x 的宽度维度进行线性变换
        x_w = self.proj_w(x)
        # 保留原始输入 x 作为第三个输入
        x_id = x
        # 将三个输入沿着通道维度拼接在一起
        x_fuse = torch.cat([x_h, x_w, x_id], dim=1)
        # 通过融合层进行处理
        out = self.fuse(x_fuse.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


# 程序入口点
if __name__ == '__main__':
    # 创建一个随机输入张量，模拟输入数据
    input = torch.randn(50, 3, 224, 224)
    # 实例化 sMLPBlock 模块
    smlp = sMLPBlock(h=224, w=224)
    # 将输入数据传递给 sMLPBlock 模块进行前向传播
    out = smlp(input)
    # 打印输出张量的形状
    print(out.shape)