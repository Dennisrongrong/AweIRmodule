import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []

        # 多个卷积层
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))  # ReLU 激活函数
            in_channels = out_channels  # 更新输入通道数

        # 池化层
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# 示例：VGG16 的构建
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.block1 = VGGBlock(3, 64, 2)  # 第一块（输入通道为3，输出64，卷积层数量为2）
        self.block2 = VGGBlock(64, 128, 2)  # 第二块
        self.block3 = VGGBlock(128, 256, 3)  # 第三块
        self.block4 = VGGBlock(256, 512, 3)  # 第四块
        self.block5 = VGGBlock(512, 512, 3)  # 第五块

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)  # 假设1000个类别输出

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = x.view(x.size(0), -1)  # 展开成一个向量
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x