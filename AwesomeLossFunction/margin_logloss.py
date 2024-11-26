import torch
import torch.nn as nn

class MarginLogLoss(nn.Module):
    """
    带边界约束的损失函数
    """
    def __init__(self, cls, margins, ignore_index=255):
        """
        初始化损失函数
        :param cls: 类别数
        :param margins: 每个类别对应的边界值列表
        :param ignore_index: 忽略的标签索引，默认 255
        """
        super(MarginLogLoss, self).__init__()
        self.register_buffer('margins', torch.tensor(margins, dtype=torch.float32))
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        前向计算损失值
        :param logits: 模型输出，形状为 (batch_size, num_classes, H, W)
        :param target: 目标标签，形状为 (batch_size, H, W)
        :return: 计算的损失值 (标量)
        """
        n, c, h, w = logits.size()
        # 调整 logits 的形状为 (batch_size * H * W, num_classes)
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, c)
        # 展平 target 为 (batch_size * H * W,)
        target = target.view(-1)

        # 创建掩码，过滤掉 ignore_index 的像素
        mask = target != self.ignore_index
        logits = logits[mask]  # 只保留有效像素
        target = target[mask]  # 只保留有效像素的目标

        # 将目标标签转换为 one-hot 编码
        one_hot = torch.zeros_like(logits, device=logits.device)  # 初始化 one-hot 编码
        one_hot.scatter_(1, target.unsqueeze(1), 1.0)  # 填充 one-hot 编码

        # 获取对应类别的边界值
        margins = self.margins[target]  # (有效像素数,)
        margins_one_hot = one_hot * margins.unsqueeze(1)  # 将边界值扩展到 one-hot 编码

        # 计算 binary_cross_entropy_with_logits 损失
        loss = nn.functional.binary_cross_entropy_with_logits(logits, margins_one_hot, reduction='mean')
        return loss
