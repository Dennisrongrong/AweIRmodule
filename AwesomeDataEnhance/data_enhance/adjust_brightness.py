import random
from PIL import Image, ImageEnhance

def adjust_brightness_random(image, min_factor=0.3, max_factor=3):
    """
    随机调整图像亮度。

    Args:
        image (PIL.Image): 输入图像。
        min_factor (float): 最小亮度因子。
        max_factor (float): 最大亮度因子。

    Returns:
        PIL.Image: 亮度调整后的图像。
        float: 随机生成的亮度因子。
    """
    factor = random.uniform(min_factor, max_factor)  # 生成随机亮度因子
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor), factor
