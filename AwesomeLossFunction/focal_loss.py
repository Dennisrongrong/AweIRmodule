import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss，用于处理类别不平衡问题
    参考论文: 'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss = -alpha * (1 - pt)^gamma * log(pt)

    参数:
    - apply_nonlin: 可选的非线性函数（如Softmax）。
    - alpha: 类别平衡因子，可以是标量、列表或张量。
    - gamma: 控制易分样本的惩罚力度，默认值为2。
    - balance_index: 用于设置 alpha 的类别平衡索引（如果 alpha 是标量）。
    - smooth: 平滑因子，防止数值不稳定。
    - size_average: 是否对损失求平均。
    """
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None and (self.smooth < 0 or self.smooth > 1.0):
            raise ValueError("smooth 应在 [0, 1] 范围内")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)

        num_class = logit.shape[1]

        if logit.dim() > 2:
            # 展平 logit
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = target.view(-1, 1)

        # 处理 alpha
        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError("不支持的 alpha 类型")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        # 计算 one-hot 编码
        idx = target.cpu().long()
        one_hot_key = torch.zeros(target.size(0), num_class).to(logit.device)
        one_hot_key.scatter_(1, idx, 1)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt

        return loss.mean() if self.size_average else loss.sum()


class FocalLossV2(nn.Module):
    """
    Focal Loss 的改进版本，支持更多自定义功能
    继承自 FocalLoss 类，保持接口一致
    """
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLossV2, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None and (self.smooth < 0 or self.smooth > 1.0):
            raise ValueError("smooth 应在 [0, 1] 范围内")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)

        num_class = logit.shape[1]

        if logit.dim() > 2:
            # 展平 logit
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = target.view(-1, 1)

        # 处理 alpha
        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError("不支持的 alpha 类型")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        # 计算 one-hot 编码
        idx = target.cpu().long()
        one_hot_key = torch.zeros(target.size(0), num_class).to(logit.device)
        one_hot_key.scatter_(1, idx, 1)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt

        return loss.mean() if self.size_average else loss.sum()
