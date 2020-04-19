import tensorflow as tf
from tensorflow.keras import backend as K
from configs.config import CONFIG
from utils.get_img_middle_layer import GetImgMiddleLayer

def loss_sum(content_and_style_matrix):
    pass

def loss_content(a, b=GetImgMiddleLayer().content_target()):
    pass

def loss_style(a, b=GetImgMiddleLayer().style_target()):
    pass

style_targets = GetImgMiddleLayer().style_target()
content_targets=GetImgMiddleLayer().content_target()
style_weight=CONFIG.alpha
content_weight=CONFIG.beta
num_style_layers = len(CONFIG.style_layer)
num_content_layers = len(CONFIG.content_layer)
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss