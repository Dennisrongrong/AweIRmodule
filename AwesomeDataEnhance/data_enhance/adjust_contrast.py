import random
from PIL import Image, ImageEnhance

def adjust_contrast_random(image, min_factor=0.5, max_factor=1.5):
    """
    随机调整图像对比度。

    Args:
        image (PIL.Image): 输入图像。
        min_factor (float): 最小对比度因子。
        max_factor (float): 最大对比度因子。

    Returns:
        PIL.Image: 对比度调整后的图像。
        float: 随机生成的对比度因子。
    """
    factor = random.uniform(min_factor, max_factor)  # 生成随机对比度因子
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor), factor


