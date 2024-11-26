import numpy as np
import torch
from torch import nn
from torch.nn import init

def XNorm(x, gamma):
    """
    Normalization function that scales the input tensor x by a learnable gamma parameter.
    :param x: Input tensor
    :param gamma: Learnable scaling parameter
    :return: Scaled tensor
    """
    norm_tensor = torch.norm(x, p=2, dim=-1, keepdim=True)
    return x * gamma / norm_tensor

class UFOAttention(nn.Module):
    """
    UFOAttention module, a type of attention mechanism that uses a learnable scaling parameter.
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        """
        Initializes the UFOAttention module.
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param dropout: Dropout rate
        """
        super(UFOAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.randn((1, h, 1, 1)))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the module.
        """
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

    def forward(self, queries, keys, values):
        """
        Defines the forward pass of the UFOAttention module.
        :param queries: Queries tensor
        :param keys: Keys tensor
        :param values: Values tensor
        :return: Output tensor
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        kv = torch.matmul(k, v)  # (b_s, h, nk, d_v)
        kv_norm = XNorm(kv, self.gamma)  # (b_s, h, nk, d_v)
        q_norm = XNorm(q, self.gamma)  # (b_s, h, nq, d_k)
        out = torch.matmul(q_norm, kv_norm.permute(0, 1, 3, 2))  # (b_s, h, nq, d_v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out

if __name__ == '__main__':
    input = torch.randn(50, 49, 512)
    ufo = UFOAttention(d_model=512, d_k=64, d_v=64, h=8)  # Adjusted d_k and d_v to be divisors of d_model
    output = ufo(input, input, input)
    print(output.shape)