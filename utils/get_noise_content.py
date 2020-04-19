import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from configs.config import CONFIG


# todo 暂时还没有添加噪音
def noise_content():
    content = load_img(CONFIG.end_layer_name, target_size=(CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH))
    content = img_to_array(content)
    content = content[np.newaxis]
    return content

