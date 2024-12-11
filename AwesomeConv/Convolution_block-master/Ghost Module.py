import torch
import torch.nn as nn
import torch.nn.functional as F

# Ghost 模块
class GhostModule(nn.Module):
    """
    Ghost 模块：通过标准卷积生成主要特征（primary features），
    然后使用更简单的操作（如深度卷积）生成额外特征（ghost features）。
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, ratio=2, dw_kernel_size=3, activation=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels  # 输出通道数
        self.primary_channels = int(out_channels / ratio)  # 主卷积层输出通道数
        self.cheap_channels = out_channels - self.primary_channels  # 便宜操作输出通道数

        # 主卷积操作
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.primary_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(self.primary_channels),
            nn.ReLU(inplace=True) if activation else nn.Identity()  # 是否激活
        )

        # 便宜操作（深度可分离卷积）
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.primary_channels, self.cheap_channels, dw_kernel_size, 1, dw_kernel_size // 2,
                      groups=self.primary_channels, bias=False),
            nn.BatchNorm2d(self.cheap_channels),
            nn.ReLU(inplace=True) if activation else nn.Identity()
        )

    def forward(self, x):
        # 通过主卷积生成主特征
        primary_features = self.primary_conv(x)
        # 通过便宜操作生成附加特征
        cheap_features = self.cheap_operation(primary_features)
        # 拼接主特征和附加特征
        return torch.cat([primary_features, cheap_features], dim=1)

# Ghost 瓶颈模块
class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck：包含两个 Ghost 模块和一个可选的 SE（Squeeze-and-Excite）模块。
    """
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # 第一个 Ghost 模块
        self.ghost1 = GhostModule(in_channels, hidden_channels, kernel_size=1)

        # 深度卷积层（仅在 stride > 1 时使用）
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, kernel_size // 2,
                                   groups=hidden_channels, bias=False) if stride > 1 else nn.Identity()
        self.bn_depthwise = nn.BatchNorm2d(hidden_channels) if stride > 1 else nn.Identity()

        # 可选的 Squeeze-and-Excitation（SE）模块
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化
                nn.Conv2d(hidden_channels, hidden_channels // 4, kernel_size=1, stride=1),  # 压缩
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels // 4, hidden_channels, kernel_size=1, stride=1),  # 恢复
                nn.Sigmoid()  # 输出权重
            )

        # 第二个 Ghost 模块
        self.ghost2 = GhostModule(hidden_channels, out_channels, kernel_size=1, activation=False)

        # Shortcut 连接，用于残差连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride > 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # Shortcut 残差分支
        residual = self.shortcut(x)

        # 主路径
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.depthwise(x)
            x = self.bn_depthwise(x)

        # 如果使用 SE 模块，则通过 SE 加权特征
        if self.use_se:
            se_weight = self.se(x)
            x = x * se_weight

        x = self.ghost2(x)
        # 主路径和残差路径相加
        return x + residual

# GhostNet 主体网络
class GhostNet(nn.Module):
    """
    GhostNet 网络结构：包含输入层、Ghost 瓶颈模块和输出分类层。
    """
    def __init__(self, num_classes=1000, width=1.0):
        super(GhostNet, self).__init__()
        # 配置每个瓶颈模块的参数：(k, exp_size, c, SE, stride)
        self.cfgs = [
            [3, 16, 16, False, 1],
            [3, 48, 24, False, 2],
            [3, 72, 24, False, 1],
            [5, 72, 40, True, 2],
            [5, 120, 40, True, 1],
            [3, 240, 80, False, 2],
            [3, 200, 80, False, 1],
            [3, 184, 80, False, 1],
            [3, 184, 80, False, 1],
            [3, 480, 112, True, 1],
            [3, 672, 112, True, 1],
            [5, 672, 160, True, 2],
            [5, 960, 160, False, 1],
            [5, 960, 160, True, 1]
        ]

        # 输入卷积层
        output_channels = int(16 * width)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        # 构建 Ghost 瓶颈模块
        layers = []
        input_channels = output_channels
        for k, exp_size, c, se, s in self.cfgs:
            output_channels = int(c * width)
            hidden_channels = int(exp_size * width)
            layers.append(GhostBottleneck(input_channels, hidden_channels, output_channels, k, s, se))
            input_channels = output_channels
        self.bottlenecks = nn.Sequential(*layers)

        # 最后的卷积层
        output_channels = int(960 * width)
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        # 全局平均池化和分类层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(output_channels, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 测试代码
if __name__ == "__main__":
    model = GhostNet(num_classes=1000)
    print(model)

    # 输入一个 224x224 的测试图像
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    model = GhostModule(3,64)
    input_tensor = torch.randn(1, 3, 224, 224)  # 假设输入大小为 [batch_size, channels, height, width]
    output = model(input_tensor)
    print(output.shape)  # 输出的预测结果