import random
from PIL import Image

def crop_image_random(image, min_crop_ratio=0.5, max_crop_ratio=0.9):
    """
    随机裁剪图像。

    Args:
        image (PIL.Image): 输入图像。
        min_crop_ratio (float): 裁剪区域的最小比例（相对于图像原尺寸）。
        max_crop_ratio (float): 裁剪区域的最大比例（相对于图像原尺寸）。

    Returns:
        PIL.Image: 裁剪后的图像。
        tuple: 裁剪区域的坐标 (left, upper, right, lower)。
    """
    width, height = image.size

    # 随机生成裁剪区域的宽高比例
    crop_ratio = random.uniform(min_crop_ratio, max_crop_ratio)
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)

    # 随机选择左上角的起点坐标
    left = random.randint(0, width - crop_width)
    upper = random.randint(0, height - crop_height)

    # 计算右下角坐标
    right = left + crop_width
    lower = upper + crop_height

    # 裁剪图像
    cropped_image = image.crop((left, upper, right, lower))
    return cropped_image, (left, upper, right, lower)

