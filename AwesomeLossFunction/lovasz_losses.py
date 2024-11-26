"""
Lovasz-Softmax 和 Jaccard hinge loss 的 PyTorch 实现
作者: Maxim Berman, 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import ifilterfalse
except ImportError:  # Python 3
    from itertools import filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    计算 Lovasz 扩展的梯度
    参考论文算法 1
    :param gt_sorted: 按误差排序后的目标值 (tensor)
    :return: 梯度值 (tensor)
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # 处理仅有1像素的情况
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    计算二值分割的 IoU（交并比）
    :param preds: 预测结果 (tensor)
    :param labels: 标签 (tensor)
    :param EMPTY: 当没有交集时返回的默认值
    :param ignore: 忽略的类别
    :param per_image: 是否逐图像计算
    :return: IoU 百分比
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # 若 per_image=True，取所有图像的平均值
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    计算每个类别的 IoU 数组
    :param preds: 预测结果 (tensor)
    :param labels: 标签 (tensor)
    :param C: 类别数量
    :param EMPTY: 没有交集时返回的默认值
    :param ignore: 忽略的类别
    :param per_image: 是否逐图像计算
    :return: 每个类别的 IoU 数组 (numpy array)
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # 忽略的类别不计算
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # 若 per_image=True，取每个类别在所有图像的平均值
    return 100 * np.array(ious)


# --------------------------- 二值损失函数 ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    二值分割的 Lovasz hinge 损失
    :param logits: 模型输出的 logits (tensor)，大小为 [B, H, W]
    :param labels: 二值标签 (tensor)，大小为 [B, H, W]
    :param per_image: 是否按图像计算损失
    :param ignore: 忽略的类别
    :return: 损失值 (标量)
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    二值分割的 Lovasz hinge 损失（扁平化版本）
    :param logits: 模型输出的 logits (tensor)，大小为 [P]
    :param labels: 二值标签 (tensor)，大小为 [P]
    :return: 损失值 (标量)
    """
    if len(labels) == 0:
        # 若仅包含忽略像素，梯度应为 0
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
    扁平化二值预测和标签
    :param scores: 预测的 logits (tensor)
    :param labels: 二值标签 (tensor)
    :param ignore: 忽略的类别
    :return: 有效的预测值和标签
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    """
    稳定版本的二值交叉熵损失
    """
    def __init__(self):
         super(StableBCELoss, self).__init__()

    def forward(self, input, target):
         neg_abs = -input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


# --------------------------- 多类损失函数 ---------------------------

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    多类分割的 Lovasz Softmax 损失
    :param probas: 预测的概率 (tensor)，大小为 [B, C, H, W]
    :param labels: 多类标签 (tensor)，大小为 [B, H, W]
    :param classes: 计算的类别范围，默认为 'present'（标签中出现的类别）
    :param per_image: 是否按图像计算损失
    :param ignore: 忽略的类别
    :return: 损失值 (标量)
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    多类分割的 Lovasz Softmax 损失（扁平化版本）
    :param probas: 预测的概率 (tensor)，大小为 [P, C]
    :param labels: 多类标签 (tensor)，大小为 [P]
    :param classes: 计算的类别范围，默认为 'present'
    :return: 损失值 (标量)
    """
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # 类别 c 的前景
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid 输出仅适用于 1 个类别')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    扁平化多类预测概率和标签
    :param probas: 预测的概率 (tensor)
    :param labels: 多类标签 (tensor)
    :param ignore: 忽略的类别
    :return: 有效的预测概率和标签
    """
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


# --------------------------- 辅助函数 ---------------------------

def isnan(x):
    """
    检查值是否为 NaN
    """
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    计算列表的平均值，支持忽略 NaN 值
    :param l: 输入列表
    :param ignore_nan: 是否忽略 NaN 值
    :param empty: 列表为空时的返回值
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
