import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from AwesomeLossFunction.focal_loss import FocalLoss
from AwesomeLossFunction.soft_dice_loss import DiceLoss  # 确保有 DiceLoss 的实现

class CompoundLoss(nn.Module):
    """
    复合损失的基类，用于组合多种损失函数。
    """
    def __init__(self, alpha: float = 1.0, ignore_index: int = 255):
        """
        初始化复合损失类。
        :param alpha: 正则化项的权重，默认 1.0。
        :param ignore_index: 忽略的标签索引，默认 255。
        """
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index

    def cross_entropy(self, inputs: torch.Tensor, labels: torch.Tensor, mode: str = "multiclass"):
        """
        计算交叉熵损失。
        :param inputs: 模型输出 (batch_size, num_classes, H, W)。
        :param labels: 目标标签 (batch_size, H, W)。
        :param mode: 分割模式，支持 "multiclass" 或 "binary"。
        :return: 交叉熵损失值。
        """
        if mode == "multiclass":
            return F.cross_entropy(inputs, labels, ignore_index=self.ignore_index)
        else:
            return F.binary_cross_entropy_with_logits(inputs, labels.float())

    def get_region_proportion(self, labels: torch.Tensor, predictions: torch.Tensor):
        """
        计算区域比例（区域大小占比）。
        :param labels: 目标标签 (batch_size, H, W)。
        :param predictions: 模型输出 (batch_size, num_classes, H, W)。
        :return: 目标区域比例和预测区域比例。
        """
        valid_mask = labels >= 0  # 有效区域的掩码
        valid_mask = valid_mask.unsqueeze(1)  # 调整形状为 (batch_size, 1, H, W)
        gt_region = labels.unsqueeze(1).float() * valid_mask.float()  # 目标区域
        pred_region = F.softmax(predictions, dim=1).float() * valid_mask.float()  # 预测区域
        return gt_region, pred_region


class CrossEntropyWithL1(CompoundLoss):
    """
    交叉熵与基于区域大小差异的 L1 正则化。
    """
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        前向计算损失值。
        :param inputs: 模型输出 (batch_size, num_classes, H, W)。
        :param labels: 目标标签 (batch_size, H, W)。
        :return: 总损失值 (标量)。
        """
        loss_ce = self.cross_entropy(inputs, labels)  # 交叉熵损失
        gt_region, pred_region = self.get_region_proportion(labels, inputs)  # 区域比例
        loss_l1 = (pred_region - gt_region).abs().mean()  # L1 正则化
        return loss_ce + self.alpha * loss_l1


class FocalWithDice(CompoundLoss):
    """
    Focal 损失与 Dice 损失的组合。
    """
    def __init__(self, alpha: float = 1.0, ignore_index: int = 255):
        """
        初始化 FocalWithDice。
        :param alpha: Dice 损失的权重，默认 1.0。
        :param ignore_index: 忽略的标签索引，默认 255。
        """
        super().__init__(alpha, ignore_index)
        self.focal_loss = FocalLoss(mode="multiclass", ignore_index=self.ignore_index)  # Focal 损失
        self.dice_loss = DiceLoss(mode="multiclass", ignore_index=self.ignore_index)  # Dice 损失

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        前向计算损失值。
        :param inputs: 模型输出 (batch_size, num_classes, H, W)。
        :param labels: 目标标签 (batch_size, H, W)。
        :return: 总损失值 (标量)。
        """
        loss_focal = self.focal_loss(inputs, labels)  # Focal 损失
        loss_dice = self.dice_loss(inputs, labels)  # Dice 损失
        return loss_focal + self.alpha * loss_dice


class CrossEntropyWithKL(CompoundLoss):
    """
    交叉熵与 KL 散度正则化的组合。
    """
    def kl_div(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        计算 KL 散度。
        :param p: 目标区域概率分布。
        :param q: 预测区域概率分布。
        :return: KL 散度值。
        """
        kl = p * torch.log((p + 1e-7) / (q + 1e-7))  # KL 散度公式
        return kl.sum(dim=1).mean()

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        前向计算损失值。
        :param inputs: 模型输出 (batch_size, num_classes, H, W)。
        :param labels: 目标标签 (batch_size, H, W)。
        :return: 总损失值 (标量)。
        """
        loss_ce = self.cross_entropy(inputs, labels)  # 交叉熵损失
        gt_region, pred_region = self.get_region_proportion(labels, inputs)  # 区域比例
        loss_kl = self.kl_div(gt_region, pred_region)  # KL 散度正则化
        return loss_ce + self.alpha * loss_kl
