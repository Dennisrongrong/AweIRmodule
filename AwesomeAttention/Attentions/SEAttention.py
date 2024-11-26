import numpy as np  # 导入numpy库，用于数学运算
import torch  # 导入PyTorch库，用于深度学习模型的构建
from torch import nn  # 从torch库中导入nn模块，包含神经网络构建所需的类和函数
from torch.nn import init  # 从torch.nn模块中导入init，用于权重初始化

# 定义SEAttention类，它是一个神经网络模块，用于实现Squeeze-and-Excitation注意力机制
class SEAttention(nn.Module):
    # 初始化函数，接收通道数channel和降维比例reduction作为参数
    def __init__(self, channel=512, reduction=16):
        super().__init__()  # 调用父类的初始化函数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 创建一个自适应平均池化层，用于压缩空间维度
        self.fc = nn.Sequential(  # 创建一个顺序容器，用于存放全连接层
            nn.Linear(channel, channel // reduction, bias=False),  # 第一个全连接层，用于降维
            nn.ReLU(inplace=True),  # 激活函数ReLU
            nn.Linear(channel // reduction, channel, bias=False),  # 第二个全连接层，用于恢复维度
            nn.Sigmoid()  # Sigmoid激活函数，用于输出0到1之间的权重
        )

    # 自定义权重初始化函数
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化权重
                if m.bias is not None:  # 如果有偏置项
                    init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.BatchNorm2d):  # 如果是批量归一化层
                init.constant_(m.weight, 1)  # 初始化权重为1
                init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                init.normal_(m.weight, std=0.001)  # 使用正态分布初始化权重
                if m.bias is not None:  # 如果有偏置项
                    init.constant_(m.bias, 0)  # 初始化偏置为0

    # 前向传播函数，定义SEAttention的计算流程
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入x的尺寸
        y = self.avg_pool(x).view(b, c)  # 通过自适应平均池化层，然后重塑形状
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层，然后重塑形状为原始输入的形状
        return x * y.expand_as(x)  # 将注意力权重应用到原始输入上，并返回结果

# 测试代码
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建一个随机的输入张量
    se = SEAttention(channel=512, reduction=8)  # 创建SEAttention实例
    output = se(input)  # 调用前向传播函数
    print(output.shape)  # 打印输出的形状