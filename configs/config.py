class CONFIG:
    # 内容图片的位置
    content_pic_path = "../data/content_img/6.jpg"

    # 风格图片的位置
    style_pic_path = "../data/style_img/1.jpeg"

    # 选择到vgg哪一层结束
    end_layer_name = "block5_conv3"
    IMAGE_HEIGHT = 200
    IMAGE_WIDTH = 800
    COLOR_CHANNELS = 3

    # content，style占比
    alpha = 1e2
    beta = 1e-2

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

    save_path = "../data/generator_img"

    list_content_img = ['../data/content_img/1.jpg', '../data/content_img/2.jpg', '../data/content_img/6.jpg',
                        '../data/content_img/4.jpg', '../data/content_img/3.jpg', '../data/content_img/5.jpg']
    list_style_img = ['../data/style_img/1.jpeg', '../data/style_img/6.jpeg', '../data/style_img/5.jpeg',
                      '../data/style_img/2.jpeg', '../data/style_img/4.jpeg', '../data/style_img/3.jpeg']

