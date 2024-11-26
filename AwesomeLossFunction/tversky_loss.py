import torch
import torch.nn as nn
from torch.nn.functional import softmax


def get_tp_fp_fn(inputs, targets, axes, loss_mask=None, square=False):
    """
    计算真阳性 (TP), 假阳性 (FP), 假阴性 (FN)
    :param inputs: 模型预测的概率分布
    :param targets: 目标标签 (one-hot 编码)
    :param axes: 要计算的维度 (通常是空间维度)
    :param loss_mask: 损失掩码 (可选)，用于忽略特定像素
    :param square: 是否对误差平方
    :return: TP, FP, FN
    """
    if square:
        inputs = inputs ** 2
        targets = targets ** 2

    tp = torch.sum(inputs * targets, dim=axes)  # 真阳性
    fp = torch.sum(inputs * (1 - targets), dim=axes)  # 假阳性
    fn = torch.sum((1 - inputs) * targets, dim=axes)  # 假阴性

    if loss_mask is not None:
        tp *= loss_mask
        fp *= loss_mask
        fn *= loss_mask

    return tp, fp, fn


class TverskyLoss(nn.Module):
    """
    Tversky 损失函数
    参考论文: https://arxiv.org/pdf/1706.05721.pdf
    """

    def __init__(self, smooth=1., alpha=0.3, beta=0.7, square=False):
        """
        初始化 Tversky 损失函数
        :param smooth: 平滑项，防止除零
        :param alpha: 假阳性权重，控制 FP 的影响
        :param beta: 假阴性权重，控制 FN 的影响
        :param square: 是否对误差进行平方
        """
        super(TverskyLoss, self).__init__()
        if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
            raise ValueError("alpha 和 beta 必须在 [0, 1] 之间")
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.square = square

    def forward(self, inputs, targets, loss_mask=None):
        """
        前向计算 Tversky 损失
        :param inputs: 模型输出 (logits)，大小为 (batch_size, num_classes, H, W)
        :param targets: 目标标签 (one-hot 编码或整数形式)，大小为 (batch_size, H, W) 或 (batch_size, num_classes, H, W)
        :param loss_mask: 损失掩码 (可选)，用于忽略特定像素
        :return: Tversky 损失值 (标量)
        """
        # 检查输入形状
        if inputs.size(0) != targets.size(0) or inputs.size(2) != targets.size(1) or inputs.size(3) != targets.size(2):
            raise ValueError(f"inputs 和 targets 的形状不匹配: inputs {inputs.size()}, targets {targets.size()}")

        # 如果 targets 为整数形式，则转换为 one-hot 编码
        if len(targets.size()) == 3:
            targets = torch.nn.functional.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()

        # 将 inputs 转换为概率分布
        inputs = softmax(inputs, dim=1)

        # 计算 TP, FP, FN
        axes = tuple(range(2, inputs.ndim))  # 默认在空间维度上计算
        tp, fp, fn = get_tp_fp_fn(inputs, targets, axes, loss_mask, self.square)

        # 计算 Tversky 系数
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # 返回损失值
        return 1.0 - tversky.mean()
