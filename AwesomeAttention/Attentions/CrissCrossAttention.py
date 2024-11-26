'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''
# 声明代码来源

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

# 定义一个函数来创建一个无穷大值的矩阵，用于防止对角线位置的值被错误地关注
def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    # 初始化函数
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)  # 查询的卷积
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)  # 键的卷积
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 值的卷积
        self.softmax = Softmax(dim=3)  # Softmax激活函数
        self.INF = INF  # 无穷大矩阵函数
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的参数gamma

    # 前向传播函数
    def forward(self, x):
        m_batchsize, _, height, width = x.size()  # 获取输入的尺寸
        proj_query = self.query_conv(x)  # 查询的卷积
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)  # 查询的转置和重塑
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)  # 查询的转置和重塑
        proj_key = self.key_conv(x)  # 键的卷积
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)  # 键的转置和重塑
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)  # 键的转置和重塑
        proj_value = self.value_conv(x)  # 值的卷积
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)  # 值的转置和重塑
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)  # 值的转置和重塑
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)  # 计算水平方向的能量
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)  # 计算垂直方向的能量
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))  # 将两个方向的能量合并并应用Softmax

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)  # 提取水平方向的注意力
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)  # 提取垂直方向的注意力
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)  # 应用水平方向的注意力
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)  # 应用垂直方向的注意力
        return self.gamma*(out_H + out_W) + x  # 将两个方向的结果合并并加权

if __name__ == '__main__':
    input=torch.randn(3, 64, 7, 7)  # 创建一个随机初始化的输入张量
    model = CrissCrossAttention(64)  # 创建CrissCrossAttention实例
    outputs = model(input)  # 通过CrissCrossAttention前向传播
    print(outputs.shape)  # 打印输出的形状