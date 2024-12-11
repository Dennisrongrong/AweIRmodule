import torch
import torch.nn as nn
import torchvision


# 3x3DW卷积(含激活函数)
def Conv3x3BNReLU(in_channels, out_channels, stride, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


# 3x3DW卷积(不激活函数)
def Conv3x3BN(in_channels, out_channels, stride, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  groups=groups),
        nn.BatchNorm2d(out_channels)
    )


# 1x1PW卷积(含激活函数)
def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


# 1x1PW卷积(不含激活函数)
def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels)
    )


# 划分channels: dim默认为0，但是由于channnels位置在1，所以传参为1
class HalfSplit(nn.Module):
    def __init__(self, dim=0, first_half=True):
        super(HalfSplit, self).__init__()
        self.first_half = first_half
        self.dim = dim

    def forward(self, input):
        # 对input的channesl进行分半操作
        splits = torch.chunk(input, 2, dim=self.dim)  # 由于shape=[b, c, h, w],对于dim=1，针对channels
        return splits[0] if self.first_half else splits[1]  # 返回其中的一半


# channels shuffle增加组间交流
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


# ShuffleNet的基本单元
class ShuffleNetUnits(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride

        # 如果stride = 2，由于主分支需要加上从分支的channels，为了两者加起来等于planes，所以需要先减一下
        if self.stride > 1:
            mid_channels = out_channels - in_channels
        # 如果stride = 2，mid_channels是一半，直接除以2即可
        else:
            mid_channels = out_channels // 2
            in_channels = mid_channels
            # 进行两次切分，一次接受一半，一次接受另外一半
            self.first_half = HalfSplit(dim=1, first_half=True)  # 对channels进行切半操作, 第一次分: first_half=True
            self.second_split = HalfSplit(dim=1, first_half=False)  # 返回输入的另外一半channesl，两次合起来才是完整的一份channels

        # 两个结构的主分支都是一样的，只是3x3DW卷积中的stride不一样，所以可以调用同样的self.bottleneck，stride会自动改变
        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, in_channels),  # 没有改变channels
            Conv3x3BN(in_channels, mid_channels, stride, groups),  # 升维
            Conv1x1BNReLU(mid_channels, mid_channels)  # 没有改变channels
        )

        # 结构(d)的从分支，3x3的DW卷积——>1x1卷积
        if self.stride > 1:
            self.shortcut = nn.Sequential(
                Conv3x3BN(in_channels=in_channels, out_channels=in_channels, stride=stride, groups=groups),
                Conv1x1BNReLU(in_channels, in_channels)
            )

        self.channel_shuffle = ChannelShuffle(groups)

    def forward(self, x):
        # stride = 2: 对于结构(d)
        if self.stride > 1:
            x1 = self.bottleneck(x)  # torch.Size([1, 220, 28, 28])
            x2 = self.shortcut(x)  # torch.Size([1, 24, 28, 28])
        # 两个分支作concat操作之后, 输出的channels便为224，与planes[0]值相等
        # out输出为: torch.Size([1, 244, 28, 28])

        # stride = 1: 对于结构(c)
        else:
            x1 = self.first_half(x)  # 一开始直接将channels等分两半，x1称为主分支的一半，此时的x1: channels = 112
            x2 = self.second_split(x)  # x2称为输入的另外一半channels: 此时x2:: channels = 112
            x1 = self.bottleneck(x1)  # 结构(c)的主分支处理
        # 两个分支作concat操作之后, 输出的channels便为224，与planes[0]值相等
        # out输出为: torch.Size([1, 244, 28, 28])

        out = torch.cat([x1, x2], dim=1)  # torch.Size([1, 244, 28, 28])
        out = self.channel_shuffle(out)  # ShuffleNet的精髓
        return out


class ShuffleNetV2(nn.Module):
    # shufflenet_v2_x2_0: planes = [244, 488, 976]  layers = [4, 8, 4]
    # shufflenet_v2_x1_5: planes = [176, 352, 704]  layers = [4, 8, 4]
    def __init__(self, planes, layers, groups, is_shuffle2_0, num_classes=5):
        super(ShuffleNetV2, self).__init__()
        # self.groups = 1
        self.groups = groups

        # input: torch.Size([1, 3, 224, 224])
        self.stage1 = nn.Sequential(
            # 结构图中，对于conv1与MaxPool的stride均为2
            Conv3x3BNReLU(in_channels=3, out_channels=24, stride=2, groups=1),  # torch.Size([1, 24, 112, 112])
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # torch.Size([1, 24, 56, 56])
        )

        self.stage2 = self._make_layer(24, planes[0], layers[0], True)  # torch.Size([1, 244, 28, 28])
        self.stage3 = self._make_layer(planes[0], planes[1], layers[1], False)  # torch.Size([1, 488, 14, 14])
        self.stage4 = self._make_layer(planes[1], planes[2], layers[2], False)  # torch.Size([1, 976, 7, 7])

        # 0.5x / 1x / 1.5x 输出为1024, 2x 输出为 2048
        self.conv5 = nn.Conv2d(in_channels=planes[2], out_channels=1024 * is_shuffle2_0, kernel_size=1, stride=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # torch.Size([1, 976, 1, 1])
        self.dropout = nn.Dropout(p=0.2)  # 丢失概率为0.2

        # 0.5x / 1x / 1.5x 输入为1024, 2x 输入为 2048
        self.linear = nn.Linear(in_features=1024 * is_shuffle2_0, out_features=num_classes)

        self.init_params()

    # 此处的is_stage2作用不大，以为均采用3x3的DW卷积，也就是group=1的组卷积
    def _make_layer(self, in_channels, out_channels, block_num, is_stage2):
        layers = []
        # 在ShuffleNetV2中，每个stage的第一个结构的stride均为2；此stage的其余结构的stride均为1.
        # 对于stride =2 的情况，对应结构(d): 一开始无切分操作，主分支经过1x1——>3x3——>1x1，从分支经过3x3——>1x1，两个分支作concat操作
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=2,
                                      groups=1 if is_stage2 else self.groups))

        # 对于stride = 1的情况，对应结构(c): 一开始就切分channel，主分支经过1x1——>3x3——>1x1再与shortcut进行concat操作
        for idx in range(1, 2):
            layers.append(
                ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=self.groups))
        return nn.Sequential(*layers)

    # 何凯明的方法初始化权重
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # input: torch.Size([1, 3, 224, 224])
    def forward(self, x):
        x = self.stage1(x)  # torch.Size([1, 24, 56, 56])
        x = self.stage2(x)  # torch.Size([1, 244, 28, 28])
        x = self.stage3(x)  # torch.Size([1, 488, 14, 14])
        x = self.stage4(x)  # torch.Size([1, 976, 7, 7])
        x = self.conv5(x)  # torch.Size([1, 2048, 7, 7])
        x = self.global_pool(x)  # torch.Size([1, 2048, 1, 1])
        x = x.view(x.size(0), -1)  # torch.Size([1, 2048])
        x = self.dropout(x)
        out = self.linear(x)  # torch.Size([1, 5])
        return out


def shufflenet_v2_x2_0(**kwargs):
    planes = [244, 488, 976]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes, layers, 1, 2)
    return model


def shufflenet_v2_x1_5(**kwargs):
    planes = [176, 352, 704]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes, layers, 1, 1)
    return model


def shufflenet_v2_x1_0(**kwargs):
    planes = [116, 232, 464]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes, layers, 1, 1)
    return model


def shufflenet_v2_x0_5(**kwargs):
    planes = [48, 96, 192]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes, layers, 1, 1)
    return model


if __name__ == '__main__':
    model = shufflenet_v2_x2_0()
    # model = shufflenet_v2_x1_5()
    # model = shufflenet_v2_x1_0()
    # model = shufflenet_v2_x0_5()
    # print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)

    torch.save(model.state_dict(), "shufflenet_v2_x2_0.mdl")