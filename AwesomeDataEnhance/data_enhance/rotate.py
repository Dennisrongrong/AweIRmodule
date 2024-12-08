import random
from PIL import Image

def rotate_image_random(image, min_angle=-180, max_angle=180):
    """
    随机旋转图像。

    Args:
        image (PIL.Image): 输入图像。
        min_angle (int): 最小旋转角度。
        max_angle (int): 最大旋转角度。

    Returns:
        PIL.Image: 旋转后的图像。
    """
    angle = random.uniform(min_angle, max_angle)  # 在范围内生成随机角度
    return image.rotate(angle, expand=True), angle  # expand=True 确保图像完整显示

# 示例调用
if __name__ == "__main__":
    image_path = "your_image.jpg"  # 替换为实际图像路径
    output_dir = "output_dir"  # 替换为输出目录路径

    image = Image.open(image_path)

    # 进行多次随机旋转
    for i in range(5):  # 生成 5 张不同角度的旋转图像
        rotated_image, angle = rotate_image_random(image, min_angle=-90, max_angle=90)
        output_path = f"{output_dir}/rotated_{int(angle)}.jpg"
        rotated_image.save(output_path)
        print(f"Saved rotated image with angle {angle:.2f}°: {output_path}")
