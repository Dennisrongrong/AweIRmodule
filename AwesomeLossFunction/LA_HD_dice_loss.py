import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.ndimage import distance_transform_edt as distance


def dice_loss(score, target):
    """
    计算 Dice 损失
    参数：
        score (torch.Tensor): 预测分数 [batch_size, ...]
        target (torch.Tensor): 目标标签 [batch_size, ...]
    返回值：
        torch.Tensor: Dice 损失值
    """
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def compute_dtm(img_gt, out_shape):
    """
    计算二值掩码的前景距离变换图（Distance Transform Map）。
    参数：
        img_gt (np.ndarray): Ground truth 二值掩码，形状为 [batch_size, x, y, z]
        out_shape (tuple): 输出形状
    返回值：
        np.ndarray: 前景距离变换图，形状为 out_shape
    """
    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # 遍历 batch
        for c in range(1, out_shape[1]):  # 忽略背景（假设通道 0 为背景）
            posmask = img_gt[b].astype(np.bool_)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm


def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    计算 Hausdorff 距离损失
    参数：
        seg_soft (torch.Tensor): 预测的 softmax 结果 [batch_size, 2, x, y, z]
        gt (torch.Tensor): Ground truth 标签 [batch_size, x, y, z]
        seg_dtm (torch.Tensor): 预测的距离变换图 [batch_size, 2, x, y, z]
        gt_dtm (torch.Tensor): Ground truth 距离变换图 [batch_size, 2, x, y, z]
    返回值：
        torch.Tensor: Hausdorff 距离损失
    """
    delta_s = (seg_soft[:, 1, ...] - gt.float()) ** 2
    s_dtm = seg_dtm[:, 1, ...] ** 2
    g_dtm = gt_dtm[:, 1, ...] ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.einsum('bxyz, bxyz->bxyz', delta_s, dtm)
    hd_loss = multipled.mean()
    return hd_loss
