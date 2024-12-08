import random
from PIL import Image, ImageEnhance

def adjust_sharpness_random(image, min_factor=0.5, max_factor=2.0):
    """
    随机调整图像锐度。

    Args:
        image (PIL.Image): 输入图像。
        min_factor (float): 最小锐度因子。
        max_factor (float): 最大锐度因子。

    Returns:
        PIL.Image: 锐度调整后的图像。
        float: 随机生成的锐度因子。
    """
    factor = random.uniform(min_factor, max_factor)  # 生成随机锐度因子
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor), factor
