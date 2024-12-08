import numpy as np
from PIL import Image, ImageFilter

def kernel_filter(image, kernel):
    """
    应用自定义的卷积核对图像进行滤波。

    :param image: 输入图像。
    :param kernel: 自定义卷积核。
    :return: 经过卷积核处理后的图像。
     锐化核
    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
     边缘检测核：
    kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])
      模糊核：
    kernel = np.ones((3, 3)) / 9


    """
    kernel_size = kernel.shape[0]
    kernel_flat = kernel.flatten().tolist()

    # 计算核的归一化比例
    scale = np.sum(kernel)
    if scale == 0:
        scale = 1

    # 创建 PIL 的 Kernel 滤镜
    filter = ImageFilter.Kernel(
        (kernel_size, kernel_size),
        kernel_flat,
        scale=scale,
        offset=128  # 避免产生负值，调节偏移量
    )

    # 应用滤镜
    filtered_image = image.filter(filter)
    return filtered_image
