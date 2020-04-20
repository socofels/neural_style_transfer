import tensorflow as tf
from tensorflow.keras import backend as K

from configs.config import CONFIG
from utils.get_img_middle_layer import GetImgMiddleLayer


# 输入一个字典，包括内容张量和风格矩阵。
def loss_sum(content_and_style_matrix):
    content = content_and_style_matrix["content"]
    style_matrix = content_and_style_matrix["style"]
    loss = K.sum(CONFIG.alpha * loss_content(content), CONFIG.beta * loss_style(style_matrix))
    print(f"cotentloss:{loss_content(content)}\nstyleloss:{loss_style(style_matrix)}")
    return loss


# 输入两个张量求距离
def loss_content(a, b=GetImgMiddleLayer().content_target()):
    for layer in CONFIG.content_layer:
        a_G = a[layer]
        a_c = b[layer]
        a_G = K.flatten(a_G)
        a_c = K.flatten(a_c)
        x = tf.subtract(a_G, a_c)
        x = K.square(x)
        print(f"loss content{K.sum(x)}")
        return K.sum(x)


# 输入两组风格矩阵，返回一个loss，使用tensorflow计算。
def loss_style(a, b=GetImgMiddleLayer().style_target()):
    # 对每一层进行求风格矩阵loss
    for layer in CONFIG.style_layer:
        a_G = a[layer]
        a_S = b[layer]
        n_hw = a_S[0]
        n_c =a_S[1]
        # 后面除的是个常数项，用来做归一化的。
        loss = [K.sum(tf.square(tf.subtract(a_G, a_S))) / (4 * n_c ** 2 * (n_hw) ** 2)*CONFIG.style_layer_weight[layer]]
        loss_all_layer = tf.add_n(loss)
    print(f"loss style{loss_all_layer}")
    return loss_all_layer

