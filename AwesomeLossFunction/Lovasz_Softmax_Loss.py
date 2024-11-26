import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge 损失函数，适用于二值分割任务。
    """
    def __init__(self, per_image=True, ignore=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, labels):
        """
        :param logits: 模型输出的 logits，形状为 (B, H, W)。
        :param labels: 目标标签，形状为 (B, H, W)。
        :return: Lovasz Hinge 损失值。
        """
        if self.per_image:
            loss = torch.mean(torch.stack([
                lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), self.ignore))
                for log, lab in zip(logits, labels)
            ]))
        else:
            loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, self.ignore))
        return loss


def lovasz_hinge_flat(logits, labels):
    """
    计算平坦化后的 Lovasz Hinge 损失。
    """
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    将二值预测和标签展平，移除忽略标签。
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_grad(gt_sorted):
    """
    计算 Lovasz 损失的梯度。
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
