import random
from PIL import Image


# 随机翻转函数（包括水平和垂直翻转）
def random_flip(image, **kwargs):
    """
    随机选择进行水平翻转、垂直翻转、两者翻转或者不翻转。
    :param image: 输入的PIL Image对象
    :param kwargs: 其他额外的参数（目前无使用）
    :return: 增强后的PIL Image对象
    """
    flip_type = random.choice(['none', 'horizontal', 'vertical', 'both'])  # 随机选择翻转方式

    if flip_type == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_type == 'vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip_type == 'both':
        return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return image  # 不进行任何翻转
