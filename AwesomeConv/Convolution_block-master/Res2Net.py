import torch
import torch.nn as nn
import torch.nn.functional as function
from PIL.Image import module


class Res2NetBlock(nn.Module):
    def __init__(self, inplanes, outplanes, scales=4):
        super(Res2NetBlock, self).__init__()

        if outplanes % scales != 0:  # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')

        self.scales = scales
        # 1*1的卷积层
        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, 32, 1, 1, 0),
            nn.BatchNorm2d(32)
        )
        # 3*3的卷积层，一共有3个卷积层和3个BN层
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8)
        )
        # 1*1的卷积层
        self.outconv = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input = x
        x = self.inconv(x)

        # scales个部分
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(xs[0])
        ys.append(function.relu(self.conv1(xs[1])))
        ys.append(function.relu(self.conv2(xs[2]) + ys[1]))
        ys.append(function.relu(self.conv2(xs[3]) + ys[2]))
        y = torch.cat(ys, 1)

        y = self.outconv(y)

        output = function.relu(y + input)

        return output

if __name__ =="__main__":
    module=Res2NetBlock(32,32)
    input_tensor = torch.randn(1, 32, 224, 224)
    output=module(input_tensor)
    print(output.shape)
