import torch
from torch import nn

# 定义GCModule类，继承自PyTorch的nn.Module
class GCModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # 使用1x1卷积来生成全局上下文特征
        self.conv = nn.Conv2d(channel, 1, kernel_size=1)
        # 定义softmax函数，用于计算注意力权重
        self.softmax = nn.Softmax(dim=2)
        # 定义特征转换模块，用于捕获通道间的依赖关系
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),  # 降维
            nn.LayerNorm([channel // reduction, 1, 1]),  # 层归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(channel // reduction, channel, kernel_size=1)  # 恢复维度
        )
    
    def context_modeling(self, x):
        # 获取输入特征图的维度信息
        b, c, h, w = x.shape
        input_x = x
        # 将输入特征图展平为(b, c, hw)形状
        input_x = input_x.reshape(b, c, h * w)
        # 通过1x1卷积和softmax获取全局上下文特征
        context = self.conv(x)
        context = context.reshape(b, 1, h * w).transpose(1, 2)
        # 计算全局上下文特征
        out = torch.matmul(input_x, context)
        # 将全局上下文特征恢复为原始特征图的形状(b, c, h, w)
        out = out.reshape(b, c, 1, 1)
        return out
    
    def forward(self, x):
        # 调用context_modeling函数获取全局上下文特征
        context = self.context_modeling(x)
        # 通过特征转换模块处理全局上下文特征
        y = self.transform(context)
        # 将处理后的全局上下文特征与原始特征图相加，实现特征融合
        return x + y
    
# 测试代码
if __name__ == "__main__":
    # 创建一个随机的输入张量x，模拟一个batch size为2，64通道，32x32的特征图
    x = torch.randn(2, 64, 32, 32)
    # 实例化GCModule模块，传入通道数为64
    attn = GCModule(64)
    # 通过GCModule模块处理输入x
    y = attn(x)
    # 打印输出y的形状，应该与输入x的形状相同
    print(y.shape)