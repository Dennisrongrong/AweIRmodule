from collections import OrderedDict  # 导入有序字典
import torch
from torch import nn  # 导入PyTorch的神经网络模块

# 检查输入是否存在的辅助函数
def exist(x):
    return x is not None

# 定义残差模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # 保存传入的函数

    def forward(self, x):
        return self.fn(x) + x  # 前向传播，返回函数输出与输入的和

# 定义空间门控单元
class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, len_sen):
        super().__init__()
        self.ln = nn.LayerNorm(dim)  # 层归一化
        self.proj = nn.Conv1d(len_sen, len_sen, 1)  # 一维卷积，步幅为1

        nn.init.zeros_(self.proj.weight)  # 初始化卷积权重为0
        nn.init.ones_(self.proj.bias)  # 初始化卷积偏置为1

    def forward(self, x):
        res, gate = torch.chunk(x, 2, -1)  # 将输入x在最后一个维度上分成两部分
        ### 归一化
        gate = self.ln(gate)  # 对gate部分进行层归一化
        ### 空间投影
        gate = self.proj(gate)  # 对gate进行一维卷积

        return res * gate  # 返回res与gate的逐元素乘积

# 定义gMLP模型
class gMLP(nn.Module):
    def __init__(self, num_tokens=None, len_sen=49, dim=512, d_ff=1024, num_layers=6):
        super().__init__()
        self.num_layers = num_layers  # 模型的层数
        # 如果num_tokens存在，则创建嵌入层，否则使用身份层
        self.embedding = nn.Embedding(num_tokens, dim) if exist(num_tokens) else nn.Identity()

        # 创建gMLP的多个层
        self.gmlp = nn.ModuleList([Residual(nn.Sequential(OrderedDict([
            ('ln1_%d' % i, nn.LayerNorm(dim)),  # 第一层归一化
            ('fc1_%d' % i, nn.Linear(dim, d_ff * 2)),  # 第一层全连接
            ('gelu_%d' % i, nn.GELU()),  # GELU激活函数
            ('sgu_%d' % i, SpatialGatingUnit(d_ff, len_sen)),  # 空间门控单元
            ('fc2_%d' % i, nn.Linear(d_ff, dim)),  # 第二层全连接
        ]))) for i in range(num_layers)])  # 为每一层创建Residual模块

        # 最终输出层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),  # 最后一次归一化
            nn.Linear(dim, num_tokens),  # 输出层，将特征映射到类别数
            nn.Softmax(-1)  # 使用Softmax激活函数
        )

    def forward(self, x):
        # 嵌入层
        embeded = self.embedding(x)  # 将输入x嵌入为特征向量

        # gMLP的前向传播
        y = nn.Sequential(*self.gmlp)(embeded)  # 通过所有gMLP层

        # 转到logits
        logits = self.to_logits(y)  # 通过输出层

        return logits  # 返回logits

# 测试代码
if __name__ == '__main__':
    num_tokens = 10000  # 词汇表大小
    bs = 50  # 批大小
    len_sen = 49  # 序列长度
    num_layers = 6  # 层数
    input = torch.randint(num_tokens, (bs, len_sen))  # 创建随机输入张量
    gmlp = gMLP(num_tokens=num_tokens, len_sen=len_sen, dim=512, d_ff=1024)  # 创建gMLP模型实例
    output = gmlp(input)  # 模型前向传播
    print(output.shape)  # 打印输出张量的形状