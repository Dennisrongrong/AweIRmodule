from PIL import ImageOps


def histogram_equalization(image):
    """
    对图像进行直方图均衡化。直方图均衡化对图像进行非线性拉伸，重新分配图像像素值

    :param image: 输入的 PIL 图像
    :return: 均衡化后的图像
    """
    # 转换为灰度图像
    grayscale_image = image.convert("L")
    equalized_image = ImageOps.equalize(grayscale_image)
    return equalized_image.convert("RGB")  # 转换回 RGB
