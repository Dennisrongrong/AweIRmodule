import numpy as np
import os
import random
from PIL import Image
import rotate
import adjust_brightness
import adjust_contrast
import adjust_sharpness
import crop
import add_noise
import erasing
import cutmix
import flip
import perspective_transform
import histogram_equalization
import pca_jittering
import kernel_filter
import gamma_correction

# 定义增强函数和参数，每个增强方法会有随机化的参数
augmentations = [
    ("Rotated", rotate.rotate_image_random, {"min_angle": -45, "max_angle": 45}),
    ("Brightened", adjust_brightness.adjust_brightness_random, {"min_factor": 0.5, "max_factor": 2.0}),
    ("Contrasted", adjust_contrast.adjust_contrast_random, {"min_factor": 0.5, "max_factor": 1.5}),
    ("Sharpened", adjust_sharpness.adjust_sharpness_random, {"min_factor": 0.5, "max_factor": 2.0}),
    ("Cropped", crop.crop_image_random, {"min_crop_ratio": 0.5, "max_crop_ratio": 0.9}),
    ("Noisy", add_noise.add_gaussian_noise, {"mean": 0, "std": 25}),
    ("RandomErased", erasing.random_erasing, {}),
    ("CutMix", cutmix.cutmix, {"alpha": 1.0, "target_size": (1080, 1620)}),  # Add CutMix
    ("RandomFlipped", flip.random_flip, {}),
    ("Perspective", perspective_transform.perspective_transform, {"max_warp": 0.4}),
    ("Equalized", histogram_equalization.histogram_equalization, {}),
    ("PCAJittered", pca_jittering.pca_jittering, {"alpha_std": 3}),
    ("KernelFiltered", kernel_filter.kernel_filter, {"kernel":
         np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])}),
    ("GammaCorrected", gamma_correction.gamma_correction, {"gamma": 2.5}),
]

def process_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for file in files:
        image_path = os.path.join(input_dir, file)
        image = Image.open(image_path)

        for name, func, kwargs in augmentations:
            # 如果是 CutMix，需要传入两张图片
            if name == "CutMix":
                # 随机选择另一张图片进行混合，确保不选择相同的图片
                image2_path = random.choice([f for f in files if f != file])
                image2 = Image.open(os.path.join(input_dir, image2_path))

                augmented_image, lam = func(image, image2, **kwargs)  # 调用 CutMix 增强

                filename_params = [name, f"mixwith_{os.path.splitext(image2_path)[0]}"]
            else:
                result = func(image, **kwargs)  # 调用其他增强方法
                # 如果返回的是单个值，则直接赋值
                if isinstance(result, tuple):
                    augmented_image, *extra = result  # 拆包
                else:
                    augmented_image = result  # 直接赋值

                filename_params = [name]

            # 生成带操作名称的文件名
            save_path = os.path.join(output_dir, f"{'_'.join(filename_params)}_{file}")
            augmented_image.save(save_path)  # 保存增强后的图像
            print(f"Saved {name} image: {save_path}")



