
import torch
import torch.nn as nn

# 定义全局上下文块类
class GlobalContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type="att", fusion_types=('channel_mul')) -> None:
        super().__init__()
        # 定义有效的融合类型
        valid_fusion_types = ['channel_add', 'channel_mul']
        # 断言池化类型为'avg'或'att'
        assert pooling_type in ['avg', 'att']
        # 断言至少使用一种融合方式
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        # 初始化基本参数
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            # 否则，使用自适应平均池化
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 如果池化类型为'att'，使用1x1卷积作为掩码，并使用Softmax进行归一化
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        # 如果融合类型包含'channel_mul'，定义通道相乘卷积
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        # 定义空间池化函数
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width) # 使用1x1卷积生成掩码
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x) # 使用1x1卷积生成掩码
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)# 应用Softmax进行归一化
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask) # 计算上下文
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x) # 执行自适应平均池化
        return context

    # 定义前向传播函数
    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))  # 将权重进行放大缩小
            out = out * channel_mul_term  # 与x进行相乘
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


if __name__ == "__main__":
    input = torch.randn(16, 64, 32, 32)  #生成随机数
    net = GlobalContextBlock(64, ratio=1 / 16) #还是实例化哈
    out = net(input)
    print(out.shape)
