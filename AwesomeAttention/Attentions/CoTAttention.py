import numpy as np
import torch
from torch import nn
from torch.nn import init

class ScaledDotProductAttention(nn.Module):
    '''
    缩放点积注意力机制
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: 模型的输出维度
        :param d_k: 查询（queries）和键（keys）的维度
        :param d_v: 值（values）的维度
        :param h: 注意力头的数量
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)  # 查询的线性变换
        self.fc_k = nn.Linear(d_model, h * d_k)  # 键的线性变换
        self.fc_v = nn.Linear(d_model, h * d_v)  # 值的线性变换
        self.fc_o = nn.Linear(h * d_v, d_model)  # 输出的线性变换
        self.dropout = nn.Dropout(dropout)  # Dropout层

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()  # 初始化权重

    def init_weights(self):
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
        前向传播
        :param queries: 查询 (b_s, nq, d_model)
        :param keys: 键 (b_s, nk, d_model)
        :param values: 值 (b_s, nk, d_model)
        :param attention_mask: 注意力掩码 (b_s, h, nq, nk)。True表示掩码。
        :param attention_weights: 注意力权重 (b_s, h, nq, nk)。
        :return: 输出
        '''
        b_s, nq = queries.shape[:2]  # 获取查询的批次大小和数量
        nk = keys.shape[1]  # 获取键的数量

        # 线性变换并分割成多个头
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # 计算点积并缩放
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)  # 应用softmax
        att = self.dropout(att)  # 应用dropout

        # 计算输出
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

if __name__ == '__main__':
    input = torch.randn(50, 49, 512)  # 创建一个随机初始化的输入张量
    sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)  # 创建注意力机制实例
    output = sa(input, input, input)  # 通过注意力机制前向传播
    print(output.shape)  # 打印输出的形状