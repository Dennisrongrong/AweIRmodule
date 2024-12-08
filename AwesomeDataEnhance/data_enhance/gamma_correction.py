import numpy as np
from PIL import Image

def gamma_correction(image, gamma=1.0):
    """
    对图像应用 Gamma 校正：非线性灰度变换

    :param image: PIL.Image 对象
    :param gamma: Gamma 值。gamma > 1 提高亮度，gamma < 1 降低亮度。
    :return: 经 Gamma 校正后的 PIL.Image 对象
    """
    if not isinstance(image, Image.Image):
        raise ValueError("输入图像必须是 PIL.Image 类型")

    # 将图像转换为 NumPy 数组
    np_image = np.asarray(image, dtype=np.float32) / 255.0
    # 应用 Gamma 校正
    gamma_corrected = np.power(np_image, gamma)
    # 转换回 0-255 范围
    gamma_corrected = (gamma_corrected * 255).clip(0, 255).astype(np.uint8)
    # 转换回 PIL.Image
    return Image.fromarray(gamma_corrected)
