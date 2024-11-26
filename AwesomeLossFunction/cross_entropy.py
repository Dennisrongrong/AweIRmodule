import torch
import torch.nn as nn
from typing import Optional


class CrossEntropyLoss2d(nn.Module):
    """
    二维交叉熵损失函数，用于处理语义分割任务。
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: int = 255):
        """
        初始化函数

        :param weight: 可选，类别权重，用于处理类别不平衡问题。
        :param ignore_index: 忽略的类别索引，默认值为 255。
        """
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, reduction='mean', ignore_index=ignore_index)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失

        :param inputs: 模型输出，大小为 (batch_size, num_classes, H, W)。
        :param targets: 目标标签，大小为 (batch_size, H, W)。
        :return: 损失值。
        """
        # 检查输入维度
        if inputs.dim() != 4:
            raise ValueError(f"Expected 4D inputs, but got {inputs.dim()}D tensor.")
        if targets.dim() != 3:
            raise ValueError(f"Expected 3D targets, but got {targets.dim()}D tensor.")
        if inputs.size(0) != targets.size(0) or inputs.size(2) != targets.size(1) or inputs.size(3) != targets.size(2):
            raise ValueError("The dimensions of inputs and targets do not match.")

        # 计算损失
        return self.nll_loss(self.logsoftmax(inputs), targets)
