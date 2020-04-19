import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

from configs.config import CONFIG


def model(layers):
    # 使用的model
    vgg = VGG16(include_top=False, weights='imagenet')
    outputs = [vgg.get_layer(layer_name).output for layer_name in layers]
    mymodel = Model(inputs=vgg.input, outputs=outputs)
    mymodel.trainable = False
    return mymodel


def get_img(pic_path):
    max_dims = CONFIG.IMAGE_WIDTH
    img = tf.io.read_file(pic_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 输出shape
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    # 找到最长的shape
    long_dim = max(shape)
    scale = max_dims / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


# 输入(n_h, n_w, n_c)张量x，返回nc*nc矩阵
def get_style_matrix(x):
    m, n_h, n_w, n_c = x.get_shape().as_list()
    # 转为宽为nw×nh，高为nc的矩阵
    x = tf.reshape(x, [n_h * n_w, n_c])
    # 这里注意获得方式
    return tf.matmul(tf.transpose(x), x)




class GetImgMiddleLayer():
    def __init__(self):
        self.extractor = StyleContentModel(CONFIG.style_layer ,CONFIG.content_layer)

    def content_target(self):
        target = self.extractor(get_img(CONFIG.content_pic_path))["content"]
        return target

    def style_target(self):
        target = self.extractor(get_img(CONFIG.style_pic_path))["style"]
        return target



# 输入层的名字，返回一个一个字典{'content': content_dict, 'style': style_dict}，张量与风格矩阵
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = model(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    # python call方法，使类变为一个函数一样可调用
    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [get_style_matrix(style_output)
                         for style_output in style_outputs]
        # 生成字典，每层的名字对应每一层输出
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}



# 　打印图片
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)

    if title:
        plt.title(title)
    plt.show()
