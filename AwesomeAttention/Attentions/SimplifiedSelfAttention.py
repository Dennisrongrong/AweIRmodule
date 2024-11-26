import numpy as np  # 导入numpy库，用于数学运算
import torch  # 导入PyTorch库，用于深度学习模型的构建
from torch import nn  # 从torch库中导入nn模块，包含神经网络构建所需的类和函数
from torch.nn import init  # 从torch.nn模块中导入init，用于权重初始化

class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    简化版的缩放点积注意力机制。
    '''

    def __init__(self, d_model, h, dropout=.1):
        '''
        初始化函数，接收模型的输出维度d_model和注意力头的数量h以及dropout率。
        :param d_model: 模型的输出维度
        :param h: 注意力头的数量
        :param dropout: dropout率
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // h  # 计算每个头的维度
        self.d_v = d_model // h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)  # 输出的线性变换
        self.dropout = nn.Dropout(dropout)  # Dropout层

        self.init_weights()  # 初始化权重

    def init_weights(self):
        '''
        自定义权重初始化函数。
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.BatchNorm2d):  # 如果是批量归一化层
                init.constant_(m.weight, 1)  # 初始化权重为1
                init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                init.normal_(m.weight, std=0.001)  # 初始化权重为正态分布
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置为0

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        前向传播函数，定义缩放点积注意力的计算流程。
        :param queries: 查询 (b_s, nq, d_model)
        :param keys: 键 (b_s, nk, d_model)
        :param values: 值 (b_s, nk, d_model)
        :param attention_mask: 注意力掩码 (b_s, h, nq, nk)。True表示掩码。
        :param attention_weights: 注意力权重 (b_s, h, nq, nk)。
        :return: 输出
        '''
        b_s, nq = queries.shape[:2]  # 获取查询的批次大小和数量
        nk = keys.shape[1]  # 获取键的数量

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)  # 应用softmax
        att = self.dropout(att)  # 应用dropout

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out  # 返回输出

if __name__ == '__main__':
    input = torch.randn(50, 49, 512)  # 创建一个随机初始化的输入张量
    ssa = SimplifiedScaledDotProductAttention(d_model=512, h=8)  # 创建缩放点积注意力机制实例
    output = ssa(input, input, input)  # 通过缩放点积注意力机制前向传播
    print(output.shape)  # 打印输出的形状