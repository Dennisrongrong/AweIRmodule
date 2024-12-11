import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidualBlock, self).__init__()

        # 扩展层（1x1卷积扩展通道数）
        self.expansion = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, stride=1, bias=False)
        self.expansion_bn = nn.BatchNorm2d(in_channels * expansion_factor)

        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                                   kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion_factor,
                                   bias=False)
        self.depthwise_bn = nn.BatchNorm2d(in_channels * expansion_factor)

        # 逐点卷积（1x1卷积恢复通道数）
        self.pointwise = nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.use_residual = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # 扩展 -> 深度卷积 -> 逐点卷积 -> 残差
        out = self.expansion(x)
        out = self.expansion_bn(out)
        out = F.relu(out)

        out = self.depthwise(out)
        out = self.depthwise_bn(out)
        out = F.relu(out)

        out = self.pointwise(out)
        out = self.pointwise_bn(out)

        if self.use_residual:
            out = out + x  # 残差连接

        return out


# 测试倒置残差块
if __name__ == "__main__":
    block = InvertedResidualBlock(in_channels=32, out_channels=64, stride=1, expansion_factor=6)
    x = torch.randn(1, 32, 56, 56)
    output = block(x)
    print(f"输出形状: {output.shape}")
