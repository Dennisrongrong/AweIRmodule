# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/25 17:52
@Auth ： 归去来兮
@File ：DataAug.py
@IDE ：PyCharm
@Motto:花中自幼微风起
"""
import os
from PIL import Image, ImageEnhance
import numpy as np
import random
import shutil

# 增强参数
params = {
    "rotation_angle": 45,  # 旋转角度 (degrees)
    "flip_horizontal": True,  # 是否水平翻转
    "flip_vertical": False,  # 是否垂直翻转
    "brightness_factor": 1.5,  # 亮度调整系数 (1.0 表示无变化)
    "contrast_factor": 1.2,  # 对比度调整系数
    "sharpness_factor": 2.0,  # 锐化强度系数
    "crop_box": (50, 50, 200, 200),  # 裁剪区域 (left, upper, right, lower)
    "add_noise": True,  # 是否添加随机噪声
                        #copt-paste
}

# 输入和输出文件夹路径
dataset_dir = r"xxx"  # 训练集目录
output_dir = r"xxx"  # 增强后图像保存目录

def augment_image(image, params):
    augmented_images = []
    augmented_images.append(("Original", image))

    if "rotation_angle" in params:
        rotated = image.rotate(params["rotation_angle"])
        augmented_images.append((f"Rotated ({params['rotation_angle']}°)", rotated))

    if "flip_horizontal" in params and params["flip_horizontal"]:
        flipped_horizontal = image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(("Flipped Horizontal", flipped_horizontal))

    if "flip_vertical" in params and params["flip_vertical"]:
        flipped_vertical = image.transpose(Image.FLIP_TOP_BOTTOM)
        augmented_images.append(("Flipped Vertical", flipped_vertical))

    if "brightness_factor" in params:
        enhancer = ImageEnhance.Brightness(image)
        brightened = enhancer.enhance(params["brightness_factor"])
        augmented_images.append((f"Brightness (x{params['brightness_factor']})", brightened))

    if "contrast_factor" in params:
        enhancer = ImageEnhance.Contrast(image)
        contrasted = enhancer.enhance(params["contrast_factor"])
        augmented_images.append((f"Contrast (x{params['contrast_factor']})", contrasted))

    if "sharpness_factor" in params:
        enhancer = ImageEnhance.Sharpness(image)
        sharpened = enhancer.enhance(params["sharpness_factor"])
        augmented_images.append((f"Sharpness (x{params['sharpness_factor']})", sharpened))

    if "crop_box" in params:
        left, upper, right, lower = params["crop_box"]
        cropped = image.crop((left, upper, right, lower))
        augmented_images.append(("Cropped", cropped))

    if "add_noise" in params and params["add_noise"]:
        noisy_image = add_gaussian_noise(image, mean=0, std=25)  # 默认噪声参数
        augmented_images.append(("Noisy", noisy_image))

    return augmented_images


# 加载训练集中的所有图像并进行增强
def augment_images_in_dataset(dataset_dir, output_dir, params):
    # 如果输出目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取训练集目录中的所有图像文件
    image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 遍历图像文件
    for image_file in image_files:
        # 拼接图像文件的完整路径
        image_path = os.path.join(dataset_dir, image_file)

        # 打开图像
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image {image_file}: {e}")
            continue

        # 生成增强后的图像
        augmented_images = augment_image(image, params)

        # 将增强后的图像保存到输出目录
        for title, img in augmented_images:
            # 定义保存路径，确保图像不会覆盖原图像
            save_path = os.path.join(output_dir, f"{title}_{image_file}")
            img.save(save_path)
            print(f"Saved augmented image: {save_path}")

def add_gaussian_noise(image, mean=0, std=25):
    # 将图像转换为 NumPy 数组
    image_np = np.array(image)
    # 生成高斯噪声
    noise = np.random.normal(mean, std, image_np.shape)
    # 将噪声添加到图像
    noisy_image = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    # 转换回 PIL 图像
    return Image.fromarray(noisy_image)
# 对训练集中的所有图像进行增强并保存
augment_images_in_dataset(dataset_dir, output_dir, params)
