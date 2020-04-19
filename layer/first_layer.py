import tensorflow as tf
from tensorflow.keras.layers import Layer

from configs.config import CONFIG


class MyLayer(Layer):

    def __init__(self,**kwargs):
        # self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(CONFIG.IMAGE_HEIGHT,CONFIG.IMAGE_WIDTH,CONFIG.COLOR_CHANNELS),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        return tf.multiply(x,self.kernel)

    def compute_output_shape(self, input_shape):
        pass
        # return (input_shape)

class fitrst_layer(Layer):

    def __init__(self,  **kwargs):
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # 创建权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    # 逻辑
    def call(self, x):
        return tf.multiply(x,self.kernel)
    # 输出
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)