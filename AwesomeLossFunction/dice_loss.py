import torch
import torch.nn as nn

class GDiceLossV2(nn.Module):
    """
    广义Dice损失，用于分割任务
    """
    def __init__(self, smooth=1e-5):
        super(GDiceLossV2, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt):
        """
        :param net_output: 模型输出，大小为 (batch_size, num_classes, H, W)
        :param gt: 目标标签，大小为 (batch_size, H, W)
        :return: 损失值
        """
        softmax_output = nn.Softmax(dim=1)(net_output)
        intersect = (softmax_output * gt).sum(dim=(2, 3))
        union = softmax_output.sum(dim=(2, 3)) + gt.sum(dim=(2, 3))
        dice = 2 * intersect / (union + self.smooth)
        return 1 - dice.mean()
