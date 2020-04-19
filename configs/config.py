class CONFIG:
    # 内容图片的位置
    content_pic_path = "../data/content_img/city.jpeg"

    # 风格图片的位置
    style_pic_path = "../data/style_img/star_night.jpeg"

    # 选择到vgg哪一层结束
    end_layer_name = "block5_conv3"
    IMAGE_HEIGHT = 200
    IMAGE_WIDTH = 400
    COLOR_CHANNELS = 3

    # content，style占比
    alpha = 10
    beta = 0.1

    # 从VGG抽取输出层的名称
    content_layer = ["block5_conv3"]
    style_layer = ["block1_conv2",
                   "block2_conv2",
                   "block3_conv3",
                   "block4_conv3"
                   ]
    style_layer_weight = {"block1_conv2": 0.25,
                          "block2_conv2": 0.25,
                          "block3_conv3": 0.25,
                          "block4_conv3": 0.25}
    # 学习速率
    lr = 2

