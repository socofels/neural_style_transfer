import numpy as np
from matplotlib.image import imsave
from tensorflow.keras.optimizers import Adam
import time
from new_loss import style_content_loss
from utils.get_img_middle_layer import *
from IPython import display
from matplotlib import pyplot as plt
from scipy import misc

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

adam = Adam(learning_rate=CONFIG.lr)
extractor = StyleContentModel(CONFIG.style_layer, CONFIG.content_layer)



CONFIG.content_pic_path=CONFIG.content_pic_path
CONFIG.style_pic_path=CONFIG.style_pic_path

# 直接使用content图片，能够加速训练
image = tf.Variable(get_img(CONFIG.content_pic_path))
# 设定目标值
content_target =GetImgMiddleLayer().content_target()
style_target = GetImgMiddleLayer().style_target()



# 使用计算图
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
    # 计算梯度
    grad = tape.gradient(loss, image)
    # 梯度下降
    adam.apply_gradients([(grad, image)])
    # 图片修整
    image.assign(clip_0_1(image))
def show_pic(i, weight):
    if i % 100 == 0:
        np.random.seed(1)
        pic = np.random.random((224, 224, 3))
        pic = pic * weight * 255
        pic = pic.astype(int)
        plt.imshow(pic)
        plt.show()

def save_img(frame,epoch):
    save_path = CONFIG.save_path + f"/epoch{epoch}.jpeg"
    imsave(save_path, frame)

def run():
    start = time.time()
    epochs = 100
    steps_per_epoch = 200
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='')
        display.clear_output(wait=True)
        # imshow(image.read_value())
        pic = image.read_value()[0,:,:,:]
        pic=pic.numpy()
        save_img(pic,n)
        # plt.title("Train step: {}".format(step))
        # plt.show()
    end = time.time()
    print("Total time: {:.1f}".format(end - start))


if __name__ == "__main__":
    # for content in CONFIG.list_content_img:
    #     for style in CONFIG.list_style_img:
    #         CONFIG.content_pic_path = content
    #         CONFIG.style_pic_path = style
    #
    #         # 直接使用content图片，能够加速训练
    #         image = tf.Variable(get_img(CONFIG.content_pic_path))
    #         # 设定目标值
    #         content_target = GetImgMiddleLayer().content_target(content)
    #         style_target = GetImgMiddleLayer().style_target(style)
    #         filename = CONFIG.content_pic_path.split("/")[-1].split(".")[0]+"_"+CONFIG.style_pic_path.split("/")[-1].split(".")[0]
    #         run(filename)
    run()



