import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(Fire, self).__init__()

        # Squeeze: 1x1卷积
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        # Expand: 1x1卷积和3x3卷积
        self.expand_1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand_3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squeeze_activation(x)

        # Expand: 分别通过1x1和3x3卷积
        x1 = self.expand_1x1(x)
        x2 = self.expand_3x3(x)

        # 将输出通道拼接
        return torch.cat([x1, x2], 1)


if __name__ == "__main__":
    fire_module = Fire(in_channels=64, squeeze_channels=32, expand_channels=64)
    input_tensor = torch.randn(1, 64, 32, 32)
    output = fire_module(input_tensor)
    print(output.shape)  #torch.Size([1, 128, 32, 32])