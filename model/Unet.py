# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/6 13:42
@Auth ： 归去来兮
@File ：Unet_change.py
@IDE ：PyCharm
@Motto:花中自幼微风起
"""
import torch
import torch.nn as nn
import numpy as np
#from   thop import profile, clever_format
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())
class NewBlock(nn.Module):
    def __init__(self, in_channels, stride,kernel_size,padding):
        super(NewBlock, self).__init__()
        reduced_channels = int(in_channels/2)
        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.layer2 = conv_batch(reduced_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel=1, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes //1, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 1, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class Unet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block='Res_block', num_blocks=[2,2,2,2], nb_filter=[1,2,4,8,16]):
        super(Unet, self).__init__()
        if block == 'Res_block':
            block = Res_block
        if block == 'Res_CBAM_block':
            block = Res_CBAM_block
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.att_blk = eca_layer()
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
if __name__ == '__main__':
    print("是否可用：", torch.cuda.is_available())  # 查看GPU是否可用
    print("GPU数量：", torch.cuda.device_count())  # 查看GPU数量
    print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
    net = Unet()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    DATA = torch.randn(8, 1, 480, 480).to(DEVICE)
    net.cuda()
    output = net(DATA)
    print("output:", np.shape(output))
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))


# # #####################################
# # ### FLops, Params, Inference time evaluation
# if __name__ == '__main__':
#     from model.load_param_data import  load_param
#     import time
#     import os
#     from torchstat import stat
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
#     nb_filter, num_blocks= load_param('two', 'resnet_18')
#     input       = torch.randn(1, 3, 256, 256,).cuda()
#     in_channels = 3
#     # model   = res_UNet(num_classes=1, input_channels=in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter)
#     model       = LightWeightNetwork(num_classes=1, input_channels=in_channels, block=Res_block, num_blocks=num_blocks, nb_filter=nb_filter)
#     a           = stat(model, (3,256,256))
#     # model = model.cuda()
#     # flops, params = profile(model, inputs=(input,), verbose=True)
#     # flops, params = clever_format([flops, params], "%.3f")
#     # start_time = time.time()
#     # output     = model(input)
#     # end_time   = time.time()
#     # print('flops:', flops, 'params:', params)
#     # print('inference time per image:',end_time-start_time )
