import numpy as np
from PIL import Image

def add_gaussian_noise(image, mean=0, std=25):
    np_image = np.array(image)
    noise = np.random.normal(mean, std, np_image.shape).astype(np.uint8)
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# 示例调用
if __name__ == "__main__":
    image_path = "your_image.jpg"
    output_path = "noisy_image.jpg"

    image = Image.open(image_path)
    noisy_image = add_gaussian_noise(image)
    noisy_image.save(output_path)
    print(f"Saved noisy image: {output_path}")
