import random
import numpy as np
from PIL import Image


def rand_bbox(size, lamb):
    """
    生成随机的bounding box
    :param size: 图像的尺寸
    :param lamb: 用于计算bbox大小的lambda
    :return: 计算出的bounding box位置（x1, y1, x2, y2）
    """
    W = size[0]
    H = size[1]

    # 得到一个bbox和原图的比例
    cut_ratio = np.sqrt(1.0 - lamb)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    # 得到bbox的中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(image1, image2, alpha=1.0, target_size=(256, 256)):
    """
    CutMix 数据增强：随机选择一部分图像进行混合
    :param image1: 第一张图像
    :param image2: 第二张图像
    :param alpha: 控制lambda的值
    :param target_size: 输出图像的目标尺寸
    :return: 混合后的图像和标签
    """
    # 确保图像尺寸一致
    image1 = image1.resize(target_size)
    image2 = image2.resize(target_size)

    # 生成随机的lambda值
    lam = np.random.beta(alpha, alpha)
    lam = random.random() / 2 + 0.3  # 可选的范围修改，确保lam值合理

    # 获取图像的尺寸
    W, H = image1.size
    bbx1, bby1, bbx2, bby2 = rand_bbox((W, H), lam)

    # 将图像1中的bbox区域替换为图像2中的相应区域
    image1 = np.array(image1)
    image2 = np.array(image2)

    # 替换区域
    image1[bbx1:bbx2, bby1:bby2, :] = image2[bbx1:bbx2, bby1:bby2, :]

    # 根据裁剪区域的面积计算新的lambda值
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1)) / (W * H)

    return Image.fromarray(image1), lam

