import random
import numpy as np
from PIL import Image

def calculate_perspective_coefficients(src_points, dst_points):
    """
    计算透视变换矩阵的系数。

    :param src_points: 原始图像的 4 个顶点坐标 [(x1, y1), (x2, y2), ...]。
    :param dst_points: 目标图像的 4 个顶点坐标 [(x1', y1'), (x2', y2'), ...]。
    :return: 透视变换的 8 个系数列表。
    """
    matrix = []
    for (src, dst) in zip(src_points, dst_points):
        matrix.append([src[0], src[1], 1, 0, 0, 0, -dst[0] * src[0], -dst[0] * src[1]])
        matrix.append([0, 0, 0, src[0], src[1], 1, -dst[1] * src[0], -dst[1] * src[1]])
    A = np.array(matrix, dtype=np.float32)
    B = np.array([pt for pair in dst_points for pt in pair], dtype=np.float32)
    coefficients = np.linalg.solve(A, B)
    return coefficients.tolist()

def perspective_transform(image, max_warp=0.4):
    """
    对图像应用随机透视变换。

    :param image: 输入的 PIL 图像。
    :param max_warp: 最大变形系数，占图像尺寸的比例。
    :return: 应用透视变换后的 PIL 图像。
    """
    width, height = image.size

    # 生成角点的随机偏移量
    x_warp = max_warp * width
    y_warp = max_warp * height

    # 定义原始角点（图像的四个顶点）
    src_points = [
        (0, 0), (width, 0), (width, height), (0, height)
    ]

    # 定义目标角点（加入随机扰动）
    dst_points = [
        (random.uniform(-x_warp, x_warp), random.uniform(-y_warp, y_warp)),
        (width + random.uniform(-x_warp, x_warp), random.uniform(-y_warp, y_warp)),
        (width + random.uniform(-x_warp, x_warp), height + random.uniform(-y_warp, y_warp)),
        (random.uniform(-x_warp, x_warp), height + random.uniform(-y_warp, y_warp)),
    ]

    # 计算透视变换的系数
    coefficients = calculate_perspective_coefficients(src_points, dst_points)

    # 应用透视变换
    transformed_image = image.transform((width, height), Image.PERSPECTIVE, coefficients, resample=Image.BICUBIC)

    return transformed_image

# 测试代码
if __name__ == "__main__":
    # 输入图像路径
    input_image_path = r"C:\Users\lenovo\Desktop\tast_database\p1.jpeg" # 替换为你的输入图像路径
    output_image_path = "test_image_transformed.jpg"  # 替换为输出图像路径

    try:
        # 打开图片
        image = Image.open(input_image_path)

        # 应用透视变换
        transformed_image = perspective_transform(image)

        # 显示变换结果
        transformed_image.show()

        # 保存结果
        transformed_image.save(output_image_path)
        print(f"透视变换后的图像已保存到: {output_image_path}")

    except Exception as e:
        print(f"发生错误: {e}")
