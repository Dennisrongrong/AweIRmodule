import numpy as np
import torch
from torch import nn
from torch.nn import init


# 定义PSA类，它继承自PyTorch的nn.Module
class PSA(nn.Module):
    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S  # S是分支的数量

        # 创建一个卷积层列表，每个分支一个
        self.convs = []
        for i in range(S):
            self.convs.append(nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1))

        # 创建一个SE（Squeeze-and-Excitation）块列表，每个分支一个
        self.se_blocks = []
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))

        self.softmax = nn.Softmax(dim=1)  # 定义softmax层，用于计算注意力权重

    def init_weights(self):
        # 初始化权重的方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化卷积层的权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 初始化BatchNorm的权重为1
                init.constant_(m.bias, 0)  # 初始化BatchNorm的偏置为0
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 初始化线性层的权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 初始化偏置为0

    def forward(self, x):
        b, c, h, w = x.size()

        # Step1: SPC（Split and Concatenate）模块
        SPC_out = x.view(b, self.S, c // self.S, h, w)  # 将输入x重塑为[b, S, ci, h, w]
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])  # 对每个分支应用卷积

        # Step2: SE权重
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :, :]))  # 对每个分支应用SE块
        SE_out = torch.stack(se_out, dim=1)  # 将SE块的输出堆叠起来
        SE_out = SE_out.expand_as(SPC_out)  # 扩展SE块的输出以匹配SPC模块的输出尺寸

        # Step3: Softmax
        softmax_out = self.softmax(SE_out)  # 应用softmax获取注意力权重

        # Step4: SPA（Self-Attention）模块
        PSA_out = SPC_out * softmax_out  # 将SPC模块的输出与注意力权重相乘
        PSA_out = PSA_out.view(b, -1, h, w)  # 重塑输出为原始尺寸

        return PSA_out


# 测试代码
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建一个随机输入张量
    psa = PSA(channel=512, reduction=8)  # 创建PSA模块实例
    output = psa(input)  # 将输入传递给PSA模块
    a = output.view(-1).sum()  # 将输出展平并计算所有元素的和
    a.backward()  # 执行反向传播
    print(output.shape)  # 打印输出张量的形状