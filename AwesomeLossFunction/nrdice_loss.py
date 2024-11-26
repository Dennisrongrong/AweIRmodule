import torch
import torch.nn as nn

class NRDiceLoss(nn.Module):
    """
    噪声鲁棒的Dice损失
    """
    def __init__(self, gamma=1.5):
        super(NRDiceLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        :param inputs: 模型输出，大小为 (batch_size, num_classes, H, W)
        :param targets: 目标标签，大小为 (batch_size, H, W)
        :return: 损失值
        """
        inputs = nn.Softmax(dim=1)(inputs)
        numerator = torch.abs(inputs - targets).pow(self.gamma).sum(dim=(2, 3))
        denominator = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        return (numerator / (denominator + 1e-5)).mean()
