from torch import conv2d, nn  # 从PyTorch库中导入conv2d函数和nn模块
import torch  # 导入PyTorch库
from torch.nn import functional as F  # 导入PyTorch的函数模块

# 定义一个函数，用于将卷积层和批标准化层的参数转换为融合后的卷积层参数
def transI_conv_bn(conv, bn):
    std = (bn.running_var + bn.eps).sqrt()  # 计算批标准化层的标准差
    gamma = bn.weight  # 获取批标准化层的gamma参数

    weight = conv.weight * ((gamma / std).reshape(-1, 1, 1, 1))  # 将卷积层权重与标准化因子相乘
    if (conv.bias is not None):  # 如果卷积层有偏置项
        bias = gamma / std * conv.bias - gamma / std * bn.running_mean + bn.bias  # 计算融合后的偏置项
    else:
        bias = bn.bias - gamma / std * bn.running_mean  # 如果没有偏置项，则直接使用批标准化层的偏置项
    return weight, bias  # 返回融合后的权重和偏置项

# 定义一个函数，用于将两个卷积层的权重和偏置项相加
def transII_conv_branch(conv1, conv2):
    weight = conv1.weight.data + conv2.weight.data  # 将两个卷积层的权重相加
    bias = conv1.bias.data + conv2.bias.data  # 将两个卷积层的偏置项相加
    return weight, bias  # 返回相加后的权重和偏置项

# 定义一个函数，用于将两个卷积层的权重进行卷积操作
def transIII_conv_sequential(conv1, conv2):
    weight = F.conv2d(conv2.weight.data, conv1.weight.data.permute(1, 0, 2, 3))  # 将第二个卷积层的权重与第一个卷积层的权重进行卷积
    return weight  # 返回卷积后的权重

# 定义一个函数，用于将两个卷积层的权重进行拼接
def transIV_conv_concat(conv1, conv2):
    print(conv1.bias.data.shape)  # 打印第一个卷积层偏置项的形状
    print(conv2.bias.data.shape)  # 打印第二个卷积层偏置项的形状
    weight = torch.cat([conv1.weight.data, conv2.weight.data], 0)  # 将两个卷积层的权重进行拼接
    bias = torch.cat([conv1.bias.data, conv2.bias.data], 0)  # 将两个卷积层的偏置项进行拼接
    return weight, bias  # 返回拼接后的权重和偏置项

# 定义一个函数，用于创建一个平均池化的卷积层
def transV_avg(channel, kernel):
    conv = nn.Conv2d(channel, channel, kernel, bias=False)  # 创建一个卷积层，不包含偏置项
    conv.weight.data[:] = 0  # 将卷积层的权重初始化为0
    for i in range(channel):  # 对于每个输出通道
        conv.weight.data[i, i, :, :] = 1 / (kernel * kernel)  # 将权重设置为1/(kernel*kernel)，实现平均池化
    return conv  # 返回创建的平均池化卷积层

# 定义一个函数，用于将三个卷积层的权重进行融合
def transVI_conv_scale(conv1, conv2, conv3):
    weight = F.pad(conv1.weight.data, (1, 1, 1, 1)) + F.pad(conv2.weight.data, (0, 0, 1, 1)) + F.pad(conv3.weight.data, (1, 1, 0, 0))  # 对三个卷积层的权重进行填充并相加
    bias = conv1.bias.data + conv2.bias.data + conv3.bias.data  # 将三个卷积层的偏置项相加
    return weight, bias  # 返回融合后的权重和偏置项

if __name__ == '__main__':
    input = torch.randn(1, 64, 7, 7)  # 创建一个随机输入张量

    # 定义三个卷积层，分别对应1x1、1x3和3x1的卷积核
    conv1x1 = nn.Conv2d(64, 64, 1)
    conv1x3 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
    conv3x1 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
    out1 = conv1x1(input) + conv1x3(input) + conv3x1(input)  # 将三个卷积层的输出相加

    # 定义一个融合后的卷积层
    conv_fuse = nn.Conv2d(64, 64, 3, padding=1)
    conv_fuse.weight.data, conv_fuse.bias.data = transVI_conv_scale(conv1x1, conv1x3, conv3x1)  # 使用transVI_conv_scale函数融合权重和偏置项
    out2 = conv_fuse(input)  # 使用融合后的卷积层进行卷积操作

    print("difference:", ((out2 - out1) ** 2).sum().item())  # 打印两个输出之间的差异