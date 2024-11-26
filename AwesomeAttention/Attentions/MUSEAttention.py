import numpy as np
import torch
from torch import nn
from torch.nn import init

# Depth_Pointwise_Conv1d 类定义了一个深度可分离的一维卷积层
class Depth_Pointwise_Conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if k == 1:
            self.depth_conv = nn.Identity()  # 如果核大小为1，使用恒等变换
        else:
            self.depth_conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,  # 深度可分离卷积
                padding=k // 2
            )
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )

    def forward(self, x):
        out = self.pointwise_conv(self.depth_conv(x))
        return out

# MUSEAttention 类定义了一种自注意力机制
class MUSEAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)  # 查询的线性变换
        self.fc_k = nn.Linear(d_model, h * d_k)  # 键的线性变换
        self.fc_v = nn.Linear(d_model, h * d_v)  # 值的线性变换
        self.fc_o = nn.Linear(h * d_v, d_model)  # 输出的线性变换
        self.dropout = nn.Dropout(dropout)

        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)  # 深度卷积层
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)  # 深度卷积层
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)  # 深度卷积层
        self.dy_paras = nn.Parameter(torch.ones(3))  # 可学习的参数
        self.softmax = nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        # 自注意力计算
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)

        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)
        out2 = self.dy_paras[0] * self.conv1(v2) + self.dy_paras[1] * self.conv3(v2) + self.dy_paras[2] * self.conv5(v2)
        out2 = out2.permute(0, 2, 1)

        out = out + out2
        return out

if __name__ == '__main__':
    input = torch.randn(50, 49, 512)
    sa = MUSEAttention(d_model=512, d_k=512, d_v=512, h=8)
    output = sa(input, input, input)
    print(output.shape)