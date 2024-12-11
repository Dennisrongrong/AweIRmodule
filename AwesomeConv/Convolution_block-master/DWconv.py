import torch
from torch import nn

#深度可分离卷积块，与基本卷积块相比节省大量参数
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积,，使用与输入通道数相同的组数，使每个输入通道独立卷积
        self.depth_conv = nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                                  out_channels=in_channel,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  groups=in_channel),
                                        nn.BatchNorm2d(in_channels),
                                        # 激活函数层，使用LeakyReLU
                                        nn.LeakyReLU(0.1, inplace=True)
                                        )
        # 逐点卷积,使用1x1卷积核进行卷积，以改变通道数
        self.point_conv = nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1) ,
                                    nn.BatchNorm2d(out_channel),
                                    # 激活函数层，使用LeakyReLU
                                    nn.LeakyReLU(0.1, inplace=True)
                                       )


    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


# 创建模型
if __name__ == "__main__":
    in_channels = 3  # 输入通道数（例如 RGB 图像）
    out_channels = 64  # 输出通道数
    model = DepthWiseConv(in_channels, out_channels)

    # 创建一个示例输入（假设是一个批次大小为 1 的 RGB 图像，尺寸为 224x224）
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size: 1, Channels: 3, Height & Width: 224

    # 前向传播
    output = model(input_tensor)

    # 打印输出的尺寸
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    #Input shape: torch.Size([1, 3, 224, 224])
    #Output shape: torch.Size([1, 64, 224, 224])