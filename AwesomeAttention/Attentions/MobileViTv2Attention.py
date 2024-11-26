import numpy as np
import torch
from torch import nn
from torch.nn import init

class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        初始化函数，其中 d_model 是模型的输出维度。
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)  # 输入到权重的线性变换
        self.fc_k = nn.Linear(d_model, d_model)  # 输入到键的线性变换
        self.fc_v = nn.Linear(d_model, d_model)  # 输入到值的线性变换
        self.fc_o = nn.Linear(d_model, d_model)  # 输出的线性变换

        self.d_model = d_model
        self.init_weights()  # 初始化权重

    def init_weights(self):
        '''
        权重初始化函数，用于初始化模块中的权重。
        '''
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

    def forward(self, input):
        '''
        前向传播函数，计算注意力输出。
        '''
        i = self.fc_i(input)  # (bs, nq, 1)
        weight_i = torch.softmax(i, dim=1)  # 计算权重，bs,nq,1
        context_score = weight_i * self.fc_k(input)  # 计算上下文分数，bs,nq,d_model
        context_vector = torch.sum(context_score, dim=1, keepdim=True)  # 计算上下文向量，bs,1,d_model
        v = self.fc_v(input) * context_vector  # 计算值，bs,nq,d_model
        out = self.fc_o(v)  # 输出线性变换，bs,nq,d_model

        return out

if __name__ == '__main__':
    # 创建一个随机输入张量，形状为 (batch_size, sequence_length, feature_dim)。
    input = torch.randn(50, 49, 512)
    # 实例化 MobileViTv2Attention 模块，传入特征维度 d_model。
    sa = MobileViTv2Attention(d_model=512)
    # 通过模型传递输入，获取输出。
    output = sa(input)
    # 打印输出张量的形状。
    print(output.shape)