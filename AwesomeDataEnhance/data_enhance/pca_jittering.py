import numpy as np
import random
from PIL import Image


def pca_jittering(image, alpha_std=3):
    """
    对图像进行 PCA 抖动，增加色彩扰动。

    :param image: 输入的 PIL 图像。
    :param alpha_std: 控制扰动强度的标准差。
    :return: 增强后的图像。
    """
    img_array = np.array(image, dtype='float32') / 255.0  # 归一化
    height, width, channels = img_array.shape
    img_flat = img_array.reshape(-1, 3).T  # 展平并转置为 (3, H × W)

    # 计算协方差矩阵和特征值、特征向量
    img_cov = np.cov(img_flat)
    lamda, p = np.linalg.eigh(img_cov)  # 使用 np.linalg.eigh 更高效

    # 生成高斯随机扰动
    alpha = np.random.normal(0, alpha_std, size=3)
    v = np.dot(p.T, alpha * lamda)

    # 应用扰动并重构图像
    img_aug = img_flat + v[:, np.newaxis]
    img_aug = np.clip(img_aug.T.reshape(height, width, channels), 0, 1)  # 保持值在 [0, 1]

    return Image.fromarray((img_aug * 255).astype('uint8'))
