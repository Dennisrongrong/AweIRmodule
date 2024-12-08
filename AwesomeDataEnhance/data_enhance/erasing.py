import random
import numpy as np
from PIL import Image, ImageDraw


def random_erasing(image, min_area=0.02, max_area=0.4, min_aspect=0.3, max_aspect=3.3):
    """
    随机擦除：随机选择图像中的一块区域，并用随机颜色填充。

    Args:
        image (PIL.Image): 输入图像。
        min_area (float): 擦除区域的最小比例（相对于图像总面积）。
        max_area (float): 擦除区域的最大比例（相对于图像总面积）。
        min_aspect (float): 擦除区域宽高比的最小值。
        max_aspect (float): 擦除区域宽高比的最大值。

    Returns:
        PIL.Image: 执行了随机擦除的图像。
    """
    # 将图像转为RGB格式
    img = image.convert("RGB")
    img_width, img_height = img.size

    # 随机选择擦除区域的大小和宽高比
    area = random.uniform(min_area, max_area) * img_width * img_height
    aspect_ratio = random.uniform(min_aspect, max_aspect)

    # 根据宽高比计算擦除区域的宽和高
    h = int(np.sqrt(area * aspect_ratio))
    w = int(np.sqrt(area / aspect_ratio))

    # 随机选择擦除区域的左上角位置
    x1 = random.randint(0, img_width - w)
    y1 = random.randint(0, img_height - h)
    x2 = x1 + w
    y2 = y1 + h

    # 用随机颜色填充擦除区域
    color = tuple(random.randint(0, 255) for _ in range(3))  # 随机颜色
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], fill=color)

    return img
