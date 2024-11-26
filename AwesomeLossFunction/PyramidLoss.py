import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidLoss(nn.Module):
    """
    金字塔损失（Pyramid Loss）。
    该损失函数在不同的尺度下计算交叉熵损失，并将其平均，适合多尺度特征学习。
    """
    def __init__(self, num_classes=21, ignore_index=255, scales=(0.25, 0.5, 0.75, 1.0)):
        """
        初始化函数。
        参数：
            num_classes (int): 类别数。
            ignore_index (int): 忽略的标签索引。
            scales (tuple): 用于计算的尺度比例。
        """
        super(PyramidLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.scales = scales

    def forward(self, logits_4D, labels_3D):
        """
        前向计算金字塔损失。
        参数：
            logits_4D (torch.Tensor): 网络的输出 [N, C, H, W]。
            labels_3D (torch.Tensor): 目标标签 [N, H, W]。
        返回：
            torch.Tensor: 金字塔损失值。
        """
        h, w = labels_3D.shape[-2], labels_3D.shape[-1]  # 原始图像高度和宽度
        total_loss = F.cross_entropy(
            input=logits_4D,
            target=labels_3D.long(),
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        labels_4D = labels_3D.unsqueeze(dim=1).float()  # 增加通道维度，并转换为浮点数
        for scale in self.scales:
            if scale == 1.0:
                continue
            assert scale <= 1.0, "尺度比例不能大于1.0"
            now_h, now_w = int(scale * h), int(scale * w)
            # 对网络输出和标签进行插值缩放
            now_logits = F.interpolate(logits_4D, size=(now_h, now_w), mode='bilinear')
            now_labels = F.interpolate(labels_4D, size=(now_h, now_w), mode='nearest')
            # 计算缩放后的交叉熵损失
            now_loss = F.cross_entropy(
                input=now_logits,
                target=now_labels.squeeze(dim=1).long(),  # 转换回原始的标签形状
                ignore_index=self.ignore_index,
                reduction='mean'
            )
            total_loss += now_loss
        final_loss = total_loss / len(self.scales)  # 平均损失
        return final_loss
