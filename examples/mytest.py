from configs.config import CONFIG
from get_img_middle_layer import model, get_img

style_image = get_img(CONFIG.style_pic_path)
style_extractor = model(CONFIG.style_layer)
style_outputs = style_extractor(style_image*255)

#查看每层输出的统计信息
for name, output in zip(CONFIG.style_layer, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()

# -----------------


# extractor = StyleContentModel(CONFIG.style_layer, CONFIG.content_layer)
# content_img=get_img(CONFIG.content_pic_path)
# results = extractor(tf.constant(content_img))
#
# style_results = results['style']
#
# print('Styles:')
# for name, output in sorted(results['style'].items()):
#   print("  ", name)
#   print("    shape: ", output.numpy().shape)
#   print("    min: ", output.numpy().min())
#   print("    max: ", output.numpy().max())
#   print("    mean: ", output.numpy().mean())
#   print()
#
# print("Contents:")
# for name, output in sorted(results['content'].items()):
#   print("  ", name)
#   print("    shape: ", output.numpy().shape)
#   print("    min: ", output.numpy().min())
#   print("    max: ", output.numpy().max())
#   print("    mean: ", output.numpy().mean())