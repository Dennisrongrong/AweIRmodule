import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Optional


def soft_dice_score(
        output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    """
    计算Soft Dice得分
    :param output: 模型输出 (batch_size, num_classes, H, W)
    :param target: 目标标签 (batch_size, num_classes, H, W)
    :param smooth: 平滑项，默认 0.0
    :param eps: 数值稳定性参数，默认 1e-7
    :param dims: 计算维度，默认所有维度
    :return: Soft Dice得分 (标量)
    """
    assert output.size() == target.size(), "Output and target must have the same shape"
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    return (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)


class DiceLoss(nn.Module):
    """
    Dice损失函数，支持二分类、多分类和多标签分割任务
    """

    def __init__(
            self,
            smooth: float = 0.0,
            eps: float = 1e-7,
            ignore_index: Optional[int] = None,
            log_loss: bool = False,
            reduction: str = "mean",
    ):
        """
        :param smooth: 平滑项，默认 0.0
        :param eps: 数值稳定性参数，默认 1e-7
        :param ignore_index: 忽略的标签索引，默认 None
        :param log_loss: 是否返回-log(Dice) 损失，默认 False
        :param reduction: 返回的损失类型，支持 "mean", "sum", "none"，默认 "mean"
        """
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        前向计算
        :param y_pred: 模型输出 (batch_size, num_classes, H, W)
        :param y_true: 目标标签 (batch_size, H, W)
        :return: 损失值 (标量或每个类别的张量)
        """
        assert y_true.size(0) == y_pred.size(0), "Batch size of y_true and y_pred must match"

        # 转换为概率分布
        y_pred = F.softmax(y_pred, dim=1)

        # 将目标标签转换为 one-hot 编码
        num_classes = y_pred.size(1)
        y_true = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 忽略索引的处理
        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask
            y_true = y_true * mask

        # 计算 Dice 得分
        dims = (0, 2, 3)  # 在空间维度上计算
        dice_score = soft_dice_score(y_pred, y_true, smooth=self.smooth, eps=self.eps, dims=dims)

        # 返回损失
        if self.log_loss:
            loss = -torch.log(dice_score.clamp_min(self.eps))
        else:
            loss = 1.0 - dice_score

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss  # 返回每个类别的损失
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class GeneralisedDiceLoss(nn.Module):
    """
    广义Dice损失函数，适用于多类分割任务
    """

    def __init__(self, epsilon: float = 1e-7):
        """
        :param epsilon: 数值稳定性参数，默认 1e-7
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        前向计算
        :param input: 模型输出 (batch_size, num_classes, H, W)
        :param target: 目标标签 (batch_size, H, W)
        :return: 损失值 (标量)
        """
        input = F.softmax(input, dim=1)

        # 将目标标签转换为 one-hot 编码
        num_classes = input.size(1)
        target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Flatten tensors for computation
        input = input.reshape(num_classes, -1)  # 使用 reshape 替代 view
        target = target.reshape(num_classes, -1)  # 使用 reshape 替代 view

        # GDL权重
        weights = 1 / (target.sum(dim=1) ** 2).clamp(min=self.epsilon)
        intersection = (input * target).sum(dim=1) * weights
        denominator = (input + target).sum(dim=1) * weights

        dice = 2 * intersection.sum() / denominator.sum().clamp(min=self.epsilon)
        return 1.0 - dice
