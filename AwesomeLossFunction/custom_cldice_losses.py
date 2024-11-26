import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(torch.nn.Module):
    """
    软骨架化操作，用于生成图像的骨架。
    """
    def __init__(self, num_iter=40):
        """
        初始化软骨架化模块。
        :param num_iter: 骨架化的迭代次数。
        """
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        """
        执行软腐蚀操作。
        :param img: 输入图像张量，形状为 (batch_size, channels, H, W) 或 (batch_size, channels, D, H, W)。
        :return: 软腐蚀后的图像张量。
        """
        if len(img.shape) == 4:  # 2D 图像
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:  # 3D 图像
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        """
        执行软膨胀操作。
        :param img: 输入图像张量，形状为 (batch_size, channels, H, W) 或 (batch_size, channels, D, H, W)。
        :return: 软膨胀后的图像张量。
        """
        if len(img.shape) == 4:  # 2D 图像
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:  # 3D 图像
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        """
        执行软开操作（先腐蚀后膨胀）。
        :param img: 输入图像张量。
        :return: 软开操作后的图像张量。
        """
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        """
        执行软骨架化操作。
        :param img: 输入图像张量。
        :return: 骨架化后的图像张量。
        """
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, img):
        """
        前向计算骨架化。
        :param img: 输入图像张量。
        :return: 骨架化后的结果。
        """
        return self.soft_skel(img)


class soft_cldice(nn.Module):
    """
    Soft clDice 损失计算。
    """
    def __init__(self, iter_=3, smooth=1., exclude_background=False):
        """
        初始化 Soft clDice 损失。
        :param iter_: 骨架化的迭代次数。
        :param smooth: 平滑因子，防止除零。
        :param exclude_background: 是否排除背景类。
        """
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        """
        前向计算 Soft clDice 损失。
        :param y_true: 真实标签，形状为 (batch_size, channels, H, W)。
        :param y_pred: 模型预测，形状为 (batch_size, channels, H, W)。
        :return: clDice 损失值。
        """
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """
    Soft Dice 损失计算。
    :param y_true: 真实标签。
    :param y_pred: 模型预测。
    :return: Soft Dice 损失值。
    """
    smooth = 1.
    intersection = torch.sum(y_true * y_pred)
    coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return 1. - coeff


class soft_dice_cldice(nn.Module):
    """
    结合 Soft Dice 和 Soft clDice 的损失函数。
    """
    def __init__(self, iter_=3, alpha=0.5, smooth=1., exclude_background=False):
        """
        初始化 Soft Dice 和 clDice 损失。
        :param iter_: 骨架化的迭代次数。
        :param alpha: Soft Dice 与 clDice 损失的权重。
        :param smooth: 平滑因子，防止除零。
        :param exclude_background: 是否排除背景类。
        """
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.alpha = alpha
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        """
        前向计算 Soft Dice 和 clDice 损失。
        :param y_true: 真实标签。
        :param y_pred: 模型预测。
        :return: 总损失值。
        """
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        dice = soft_dice(y_true, y_pred)
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1. - self.alpha) * dice + self.alpha * cl_dice
