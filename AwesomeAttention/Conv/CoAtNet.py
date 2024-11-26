from torch import nn, sqrt  # 从PyTorch库中导入nn模块和sqrt函数
import torch  # 导入PyTorch库
import sys  # 导入系统模块
from math import sqrt  # 从math库中导入sqrt函数
sys.path.append('CoAtNet')  # 将当前目录添加到系统路径中，以便可以导入相对路径的模块
from Conv.MBConv import MBConvBlock  # 导入MBConvBlock类
from Attentions.SelfAttention import ScaledDotProductAttention  # 导入ScaledDotProductAttention类

class CoAtNet(nn.Module):  # 定义CoAtNet类，继承自PyTorch的nn.Module
    def __init__(self,in_ch,image_size,out_chs=[64,96,192,384,768]):  # 初始化函数，定义输入通道数、图像尺寸和输出通道数列表
        super().__init__()  # 调用父类的初始化函数
        self.out_chs=out_chs  # 保存输出通道数的列表
        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)  # 定义2D最大池化层
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)  # 定义1D最大池化层

        self.s0=nn.Sequential(  # 定义第一个序列模型s0
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1),  # 2D卷积层
            nn.ReLU(),  # 激活函数ReLU
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1)  # 另一个2D卷积层
        )
        self.mlp0=nn.Sequential(  # 定义第一个多层感知机mlp0
            nn.Conv2d(in_ch,out_chs[0],kernel_size=1),  # 1x1卷积层
            nn.ReLU(),  # 激活函数ReLU
            nn.Conv2d(out_chs[0],out_chs[0],kernel_size=1)  # 另一个1x1卷积层
        )

        self.s1=MBConvBlock(ksize=3,input_filters=out_chs[0],output_filters=out_chs[0],image_size=image_size//2)  # 第二阶段的MBConvBlock
        self.mlp1=nn.Sequential(  # 第二阶段的MLP
            nn.Conv2d(out_chs[0],out_chs[1],kernel_size=1),  # 1x1卷积层
            nn.ReLU(),  # 激活函数ReLU
            nn.Conv2d(out_chs[1],out_chs[1],kernel_size=1)  # 另一个1x1卷积层
        )

        self.s2=MBConvBlock(ksize=3,input_filters=out_chs[1],output_filters=out_chs[1],image_size=image_size//4)  # 第三阶段的MBConvBlock
        self.mlp2=nn.Sequential(  # 第三阶段的MLP
            nn.Conv2d(out_chs[1],out_chs[2],kernel_size=1),  # 1x1卷积层
            nn.ReLU(),  # 激活函数ReLU
            nn.Conv2d(out_chs[2],out_chs[2],kernel_size=1)  # 另一个1x1卷积层
        )

        self.s3=ScaledDotProductAttention(out_chs[2],out_chs[2]//8,out_chs[2]//8,8)  # 第四阶段的自注意力机制
        self.mlp3=nn.Sequential(  # 第四阶段的MLP
            nn.Linear(out_chs[2],out_chs[3]),  # 全连接层
            nn.ReLU(),  # 激活函数ReLU
            nn.Linear(out_chs[3],out_chs[3])  # 另一个全连接层
        )

        self.s4=ScaledDotProductAttention(out_chs[3],out_chs[3]//8,out_chs[3]//8,8)  # 第五阶段的自注意力机制
        self.mlp4=nn.Sequential(  # 第五阶段的MLP
            nn.Linear(out_chs[3],out_chs[4]),  # 全连接层
            nn.ReLU(),  # 激活函数ReLU
            nn.Linear(out_chs[4],out_chs[4])  # 另一个全连接层
        )

    def forward(self, x) :  # 前向传播函数
        B,C,H,W=x.shape  # 获取输入x的维度
        #stage0
        y=self.mlp0(self.s0(x))  # 通过s0和mlp0
        y=self.maxpool2d(y)  # 通过2D最大池化层
        #stage1
        y=self.mlp1(self.s1(y))  # 通过s1和mlp1
        y=self.maxpool2d(y)  # 通过2D最大池化层
        #stage2
        y=self.mlp2(self.s2(y))  # 通过s2和mlp2
        y=self.maxpool2d(y)  # 通过2D最大池化层
        #stage3
        y=y.reshape(B,self.out_chs[2],-1).permute(0,2,1) #B,N,C  # 调整y的形状以适应自注意力机制
        y=self.mlp3(self.s3(y,y,y))  # 通过s3和mlp3
        y=self.maxpool1d(y.permute(0,2,1)).permute(0,2,1)  # 通过1D最大池化层
        #stage4
        y=self.mlp4(self.s4(y,y,y))  # 通过s4和mlp4
        y=self.maxpool1d(y.permute(0,2,1))  # 通过1D最大池化层
        N=y.shape[-1]  # 获取y的最后一个维度的大小
        y=y.reshape(B,self.out_chs[4],int(sqrt(N)),int(sqrt(N)))  # 将y重塑为BxCxHxW的形状

        return y  # 返回输出

if __name__ == '__main__':  # 如果是主程序运行
    x=torch.randn(1,3,224,224)  # 创建一个随机初始化的输入张量
    coatnet=CoAtNet(3,224)  # 创建CoAtNet模型实例
    y=coatnet(x)  # 通过模型前向传播
    print(y.shape)  # 打印输出的形状