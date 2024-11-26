import numpy as np  # 导入numpy库，用于数学运算
import torch  # 导入PyTorch库，用于深度学习模型的构建
from torch import nn  # 从torch库中导入nn模块，包含神经网络构建所需的类和函数
from torch.nn import init  # 从torch.nn模块中导入init，用于权重初始化
from torch.nn.parameter import Parameter  # 从torch.nn模块中导入Parameter类

class ShuffleAttention(nn.Module):
    """
    洗牌注意力机制，结合通道注意力和空间注意力。
    """

    def __init__(self, channel=512, reduction=16, G=8):
        """
        初始化函数，接收通道数channel、降维比例reduction和分组数G。
        :param channel: 通道数
        :param reduction: 降维比例
        :param G: 分组数
        """
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层，用于计算通道注意力
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))  # 创建分组归一化层，用于计算空间注意力
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))  # 创建通道注意力的权重参数
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))  # 创建通道注意力的偏置参数
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))  # 创建空间注意力的权重参数
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))  # 创建空间注意力的偏置参数
        self.sigmoid = nn.Sigmoid()  # 创建Sigmoid激活函数

    def init_weights(self):
        """
        自定义权重初始化函数。
        """
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

    @staticmethod
    def channel_shuffle(x, groups):
        """
        通道洗牌函数，用于重新排列通道。
        :param x: 输入特征图
        :param groups: 分组数
        :return: 洗牌后的特征图
        """
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        """
        前向传播函数，定义洗牌注意力的计算流程。
        :param x: 输入特征图
        :return: 输出特征图
        """
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)  # bs*G,c//(2*G),h,w

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)  # b,c,h,w

        # channel shuffle
        out = self.channel_shuffle(out, 2)  # b,c,h,w
        return out  # 返回输出特征图

if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建一个随机初始化的输入张量
    se = ShuffleAttention(channel=512, G=8)  # 创建洗牌注意力模块实例
    output = se(input)  # 通过洗牌注意力模块前向传播
    print(output.shape)  # 打印输出的形状