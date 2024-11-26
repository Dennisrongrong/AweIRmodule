# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/26 12:57
@Auth ： 归去来兮
@File ：FocalLoss.py
@IDE ：PyCharm
@Motto:花中自幼微风起
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 将输入和目标展平为 (N, 1, H, W) -> (N*H*W, 1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 计算二值交叉熵损失
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 计算 p_t
        pt = torch.exp(-BCE_loss)

        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # 返回平均损失
        return focal_loss.mean()

# 示例使用
if __name__ == "__main__":
    # 假设输入和目标
    inputs = torch.randn(1,1, 256, 256, requires_grad=True)  # 输入图像
    targets = torch.randint(0, 2, (4, 3, 256, 256)).float()  # 目标掩码

    # 实例化 Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2)

    # 计算损失
    loss = focal_loss(inputs, targets)
    print("Focal Loss:", loss.item())