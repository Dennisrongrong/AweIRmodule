import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class InceptionResNetA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResNetA, self).__init__()
        # branch1: conv1*1(32)
        self.b1 = BasicConv2d(in_channels, 32, kernel_size=1)

        # branch2: conv1*1(32) --> con3*3(32)
        self.b2_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.b2_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)

        # branch3: conv1*1(32) --> conv3*3(32) --> conv3*3(32)
        self.b3_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.b3_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.b3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

        # totalbranch: conv1*1(256)
        self.tb = BasicConv2d(96, 256, kernel_size=1)

    def forward(self, x):
        x = F.relu(x)
        b_out1 = F.relu(self.b1(x))
        b_out2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b_out3 = F.relu(self.b3_3(F.relu(self.b3_2(F.relu(self.b3_1(x))))))
        b_out = torch.cat([b_out1, b_out2, b_out3], 1)
        b_out = self.tb(b_out)
        y = b_out + x
        out = F.relu(y)

        return out


class InceptionResNetA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResNetA, self).__init__()
        # branch1: conv1*1(32)
        self.b1 = BasicConv2d(in_channels, 32, kernel_size=1)

        # branch2: conv1*1(32) --> con3*3(32)
        self.b2_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.b2_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)

        # branch3: conv1*1(32) --> conv3*3(32) --> conv3*3(32)
        self.b3_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.b3_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.b3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

        # totalbranch: conv1*1(256)
        self.tb = BasicConv2d(96, 256, kernel_size=1)

    def forward(self, x):
        x = F.relu(x)
        b_out1 = F.relu(self.b1(x))
        b_out2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b_out3 = F.relu(self.b3_3(F.relu(self.b3_2(F.relu(self.b3_1(x))))))
        b_out = torch.cat([b_out1, b_out2, b_out3], 1)
        b_out = self.tb(b_out)
        y = b_out + x
        out = F.relu(y)

        return out


class InceptionResNetB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResNetB, self).__init__()
        # branch1: conv1*1(128)
        self.b1 = BasicConv2d(in_channels, 128, kernel_size=1)

        # branch2: conv1*1(128) --> con1*7(128) --> conv7*1(128)
        self.b2_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.b2_2 = BasicConv2d(128, 128, kernel_size=(1, 7), padding=(0, 3))
        self.b2_3 = BasicConv2d(128, 128, kernel_size=(7, 1), padding=(3, 0))

        # totalbranch: conv1*1(896)
        self.tb = BasicConv2d(256, 896, kernel_size=1)

    def forward(self, x):
        x = F.relu(x)
        b_out1 = F.relu(self.b1(x))
        b_out2 = F.relu(self.b2_3(F.relu(self.b2_2(F.relu(self.b2_1(x))))))
        b_out = torch.cat([b_out1, b_out2], 1)
        b_out = self.tb(b_out)
        y = b_out + x
        out = F.relu(y)

        return out


class InceptionResNetC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResNetC, self).__init__()
        # branch1: conv1*1(192)
        self.b1 = BasicConv2d(in_channels, 192, kernel_size=1)

        # branch2: conv1*1(192) --> con1*3(192) --> conv3*1(192)
        self.b2_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.b2_2 = BasicConv2d(192, 192, kernel_size=(1, 3), padding=(0, 1))
        self.b2_3 = BasicConv2d(192, 192, kernel_size=(3, 1), padding=(1, 0))

        # totalbranch: conv1*1(1792)
        self.tb = BasicConv2d(384, 1792, kernel_size=1)

    def forward(self, x):
        x = F.relu(x)
        b_out1 = F.relu(self.b1(x))
        b_out2 = F.relu(self.b2_3(F.relu(self.b2_2(F.relu(self.b2_1(x))))))
        b_out = torch.cat([b_out1, b_out2], 1)
        b_out = self.tb(b_out)
        y = b_out + x
        out = F.relu(y)

        return out


class ReductionA(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super(ReductionA, self).__init__()
        # branch1: maxpool3*3(stride2 valid)
        self.b1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # branch2: conv3*3(n stride2 valid)
        self.b2 = BasicConv2d(in_channels, n, kernel_size=3, stride=2)

        # branch3: conv1*1(k) --> conv3*3(l) --> conv3*3(m stride2 valid)
        self.b3_1 = BasicConv2d(in_channels, k, kernel_size=1)
        self.b3_2 = BasicConv2d(k, l, kernel_size=3, padding=1)
        self.b3_3 = BasicConv2d(l, m, kernel_size=3, stride=2)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = F.relu(self.b2(x))
        y3 = F.relu(self.b3_3(F.relu(self.b3_2(F.relu(self.b3_1(x))))))

        outputsRedA = [y1, y2, y3]
        return torch.cat(outputsRedA, 1)


class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        # branch1: maxpool3*3(stride2 valid)
        self.b1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # branch2: conv1*1(256) --> conv3*3(384 stride2 valid)
        self.b2_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.b2_2 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        # branch3: conv1*1(256) --> conv3*3(256 stride2 valid)
        self.b3_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.b3_2 = BasicConv2d(256, 256, kernel_size=3, stride=2)

        # branch4: conv1*1(256) --> conv3*3(256) --> conv3*3(256 stride2 valid)
        self.b4_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.b4_2 = BasicConv2d(256, 256, kernel_size=3, padding=1)
        self.b4_3 = BasicConv2d(256, 256, kernel_size=3, stride=2)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        y3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        y4 = F.relu(self.b4_3(F.relu(self.b4_2(F.relu(self.b4_1(x))))))

        outputsRedB = [y1, y2, y3, y4]
        return torch.cat(outputsRedB, 1)


class StemForIR1(nn.Module):
    def __init__(self, in_channels):
        super(StemForIR1, self).__init__()
        # conv3*3(32 stride2 valid)
        self.conv1 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        # conv3*3(32 valid)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3)
        # conv3*3(64)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        # maxpool3*3(stride2 valid)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # conv1*1(80)
        self.conv4 = BasicConv2d(64, 80, kernel_size=1)
        # conv3*3(192 valid)
        self.conv5 = BasicConv2d(80, 192, kernel_size=3)
        # conv3*3(256, stride2 valid)
        self.conv6 = BasicConv2d(192, 256, kernel_size=3, stride=2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.maxpool1(out)
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))

        return out


class InceptionResNetv1(nn.Module):
    def __init__(self):
        super(InceptionResNetv1, self).__init__()
        self.stem = StemForIR1(3)
        self.irA = InceptionResNetA(256)
        self.redA = ReductionA(256, 192, 192, 256, 384)
        self.irB = InceptionResNetB(896)
        self.redB = ReductionB(896)
        self.irC = InceptionResNetC(1792)
        self.avgpool = nn.MaxPool2d(kernel_size=8)
        self.dropout = nn.Dropout(p=0.8)
        self.linear = nn.Linear(1792, 1000)

    def forward(self, x):
        n = [5, 10, 5]
        out = self.stem(x)

        if n[0] > 0:
            out = self.irA(out)
            n[0] -= 1
        out = self.redA(out)

        if n[1] > 0:
            out = self.irB(out)
            n[1] -= 1
        out = self.redB(out)

        if n[2] > 0:
            out = self.irC(out)
            n[2] -= 1

        out = self.avgpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


# 创建并测试 InceptionResNetv1 网络
if __name__ == "__main__":
    model = InceptionResNetv1()
    input_tensor = torch.randn(1, 3, 300, 300)  # 示例输入大小,图像尺寸不能小于299
    output = model(input_tensor)
    print(output.shape)  # 输出预测结果