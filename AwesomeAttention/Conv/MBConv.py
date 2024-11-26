import math  # 导入math库，用于数学运算
from functools import partial  # 导入partial函数，用于固定函数的部分参数

import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch库中导入nn模块
from torch.nn import functional as F  # 导入PyTorch的函数式接口

# Swish激活函数的实现
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)  # 前向传播，计算Swish激活函数的值
        ctx.save_for_backward(i)  # 保存输入值，以便在反向传播时使用
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]  # 获取保存的输入值
        sigmoid_i = torch.sigmoid(i)  # 计算sigmoid函数的值
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))  # 反向传播，计算梯度

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)  # 使用SwishImplementation.apply方法来应用Swish激活函数

# Drop connect函数，一种正则化技术
def drop_connect(inputs, p, training):
    if not training: return inputs  # 如果不在训练模式，直接返回输入
    batch_size = inputs.shape[0]  # 获取批次大小
    keep_prob = 1 - p  # 计算保留概率
    random_tensor = keep_prob  # 创建一个与保留概率相同的随机张量
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)  # 生成一个随机张量
    binary_tensor = torch.floor(random_tensor)  # 将随机张量四舍五入到0或1
    output = inputs / keep_prob * binary_tensor  # 应用Drop connect
    return output

# 获取Conv2dStaticSamePadding的partial函数
def get_same_padding_conv2d(image_size=None):
    return partial(Conv2dStaticSamePadding, image_size=image_size)

# 从尺寸中获取宽度和高度
def get_width_and_height_from_size(x):
    if isinstance(x, int): return x, x  # 如果是整数，返回相同的宽度和高度
    if isinstance(x, list) or isinstance(x, tuple): return x  # 如果是列表或元组，直接返回
    else: raise TypeError()

# 计算输出图像尺寸
def calculate_output_image_size(input_image_size, stride):
    if input_image_size is None: return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]

# 定义Conv2dStaticSamePadding类，用于保持图像尺寸的卷积
class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        # 计算并保存padding值
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)  # 应用padding
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class Identity(nn.Module):  # 定义Identity模块，用于不改变输入的模块
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

# MBConvBlock类的定义
class MBConvBlock(nn.Module):
    def __init__(self, ksize, input_filters, output_filters, expand_ratio=1, stride=1, image_size=224):
        super().__init__()
        self._bn_mom = 0.1  # 设置BatchNorm的momentum参数
        self._bn_eps = 0.01  # 设置BatchNorm的eps参数
        self._se_ratio = 0.25  # 设置Squeeze-and-Excitation层的比例
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self._kernel_size = ksize
        self._stride = stride

        inp = self._input_filters
        oup = self._input_filters * self._expand_ratio
        if self._expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution
        k = self._kernel_size
        s = self._stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        Conv2d = get_same_padding_conv2d(image_size=(1,1))
        num_squeezed_channels = max(1, int(self._input_filters * self._se_ratio))
        self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._expand_ratio != 1:
            expand = self._expand_conv(inputs)
            bn0 = self._bn0(expand)
            x = self._swish(bn0)
        depthwise = self._depthwise_conv(x)
        bn1 = self._bn1(depthwise)
        x = self._swish(bn1)

        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._input_filters, self._output_filters
        if self._stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

if __name__ == '__main__':
    input=torch.randn(1,3,112,112)# 创建MBConvBlock实例
    mbconv=MBConvBlock(ksize=3,input_filters=3,output_filters=3,image_size=112)
    out=mbconv(input) # 通过MBConvBlock前向传播
    print(out.shape) # 打印输出的形状