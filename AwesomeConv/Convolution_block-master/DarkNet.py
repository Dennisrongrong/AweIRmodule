import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c_in, c_out, k, s, p, bias=True):
        '''
        自定义卷积块，一次性完成卷积+归一化+池化
        :param c_in: 输入通道数
        :param c_in: 输出通道数
        :param k:    卷积核大小
        :param s:    步长
        :param p:    填充
        :param bias: 偏置
        '''
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=bias),
            nn.BatchNorm2d(c_out, track_running_stats=False),  # 禁用 BatchNorm2d 的统计跟踪
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.conv(x)


class ConvResidual(nn.Module):
    def __init__(self, c_in):
        '''
        自定义残差单元,只需给出通道数，该单元完成两次卷积，并进行加残差后返回相同维度的特征图
        '''
        super(ConvResidual, self).__init__()
        c = c_in//2
        # 采用1*1 + 3*3 的形式加深网络深度，加强特征抽取
        self.conv = nn.Sequential(
            Conv(c_in, c, 1, 1, 0), #1x1卷积降通道
            Conv(c, c_in, 3, 1, 1), #3x3拉回通道
        )
    def forward(self, x):
        return x + self.conv(x) #残差

class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.conv1 = Conv(3, 32, 3, 1, 1)  # 一个卷积块 = 1层卷积
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.conv3_4 = ConvResidual(64) # 一个残差块 = 2层卷积   1
        self.conv5 = Conv(64, 128, 3, 2, 1)
        self.conv6_9 = nn.Sequential(    #4层卷积               2
            ConvResidual(128),
            ConvResidual(128),
        )
        self.conv10 = Conv(128, 256, 3, 2, 1)
        self.conv11_26 = nn.Sequential(*[ConvResidual(256) for i in range(8)])  # 8
        self.conv27 = Conv(256, 512, 3, 2, 1)
        self.conv28_43 = nn.Sequential(*[ConvResidual(512) for i in range(8)])  # 8
        self.conv44 = Conv(512, 1024, 3, 2, 1)
        self.conv45_52 = nn.Sequential(*[ConvResidual(1024) for i in range(4)]) # 4
        self.conv53 = Conv(1024,1000,1,1,0 )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3_4 = self.conv3_4(conv2)
        conv5 = self.conv5(conv3_4)
        conv6_9 = self.conv6_9(conv5)
        conv10 = self.conv10(conv6_9)
        conv11_26 = self.conv11_26(conv10)
        conv27 = self.conv27(conv11_26)
        conv28_43 = self.conv28_43(conv27)
        conv44 = self.conv44(conv28_43)
        conv45_52 = self.conv45_52(conv44)
        print("Before pooling:", conv45_52.shape)  # 检查卷积前的特征图
        avgpool = self.avgpool(conv45_52)
        print("Pooled shape:", avgpool.shape)  # 检查全局池化后的形状
        conv53 = self.conv53(avgpool)
        return conv53.view(x.size(0), -1)  # 展平为 [batch_size, 1000]



# 测试代码
if __name__ == "__main__":
    model = DarkNet53()
    x = torch.randn(1, 3, 416, 416)
    output = model(x)
    print("Output shape:", output.shape)
