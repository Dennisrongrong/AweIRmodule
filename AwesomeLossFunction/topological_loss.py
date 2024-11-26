import torch
import numpy as np
import gudhi as gd
from torch.nn.functional import mse_loss


def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.03):
    """
    计算预测结果与真实值之间的拓扑力向量。

    参数:
        lh_dgm: 预测结果的拓扑持久性图（二维 NumPy 数组）。
        gt_dgm: 真实值的拓扑持久性图（二维 NumPy 数组）。
        pers_thresh: 用于过滤不显著持久性值的阈值。

    返回值:
        force_list: 力向量列表。
        idx_to_fix: 修复的索引列表。
        idx_to_remove: 移除的索引列表。
        空列表（占位符）：与调用方匹配。
    """
    # 计算预测结果的持久性值（死亡时间 - 生成时间的绝对值）
    lh_pers = np.abs(lh_dgm[:, 1] - lh_dgm[:, 0])

    # 如果真实值的拓扑持久性图为空，则返回零力向量
    if gt_dgm.shape[0] == 0:
        return np.zeros_like(lh_dgm), [], list(range(len(lh_dgm))), []

    idx_to_fix = []  # 需要修复的索引
    idx_to_remove = []  # 需要移除的索引

    for i in range(len(lh_pers)):
        # 确保 lh_pers[i] 是标量
        if np.isscalar(lh_pers[i]) and lh_pers[i] > pers_thresh:  # 持久性大于阈值时需要修复
            idx_to_fix.append(i)
        else:  # 持久性小于阈值时移除
            idx_to_remove.append(i)

    # 初始化力向量列表
    force_list = np.zeros_like(lh_dgm)
    force_list[idx_to_fix, 0] = 0 - lh_dgm[idx_to_fix, 0]  # 将生成点移动到 0
    force_list[idx_to_fix, 1] = 1 - lh_dgm[idx_to_fix, 1]  # 将死亡点移动到 1

    return force_list, idx_to_fix, idx_to_remove, []  # 返回空列表占位符

def get_topo_loss(likelihood, gt, patch_size=100):
    """
    计算预测结果与真实值之间的拓扑损失。

    参数:
        likelihood: 模型的预测输出（张量）。
        gt: 真实标签（张量）。
        patch_size: 分块计算拓扑特征的大小。

    返回值:
        拓扑损失值（标量）。
    """
    # 转换为 NumPy 数组并从计算图中分离
    likelihood = torch.sigmoid(likelihood).detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    weight_map = np.zeros_like(likelihood)  # 权重图
    ref_map = np.zeros_like(likelihood)  # 参考图

    for y in range(0, likelihood.shape[0], patch_size):
        for x in range(0, likelihood.shape[1], patch_size):
            lh_patch = likelihood[y:y + patch_size, x:x + patch_size]
            gt_patch = gt[y:y + patch_size, x:x + patch_size]

            if lh_patch.max() == 0 or gt_patch.max() == 0:
                continue

            lh_diag, _, _, _ = compute_dgm_force(lh_patch, gt_patch)
            weight_map[y:y + patch_size, x:x + patch_size] = lh_diag
            ref_map[y:y + patch_size, x:x + patch_size] = gt_patch

    # 转换回 PyTorch 张量
    weight_map = torch.tensor(weight_map, dtype=torch.float32)
    ref_map = torch.tensor(ref_map, dtype=torch.float32)

    # 使用均方误差作为损失计算
    return mse_loss(weight_map, ref_map)
