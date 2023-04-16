from detection.datasets import pascal_voc
# 绘图
import matplotlib.pyplot as plt
import numpy as np
# 模型构建
from detection.models.detectors import faster_rcnn
import tensorflow as tf
# 图像展示
import visualize

from detection.datasets.utils import get_original_image

# 加载数据集
# 加载coco数据集预训练模型
# 展示RPN生成候选区域
# 展示最终候选结果

# 注意numpy版本为1.20
# matplotlib版本为3.6
# 做tensorflow实验的时候还是要注意版本问题
import matplotlib

matplotlib.rc("font", family='FangSong')

pascal = pascal_voc.pascal_voc("train")
image, imagemeta, bbox, label = pascal[218]
# 图像的均值和标准差
img_mean = (122.7717, 115.9465, 102.9801)
img_std = (1., 1., 1.)
# RGB图像
rgd_image = np.round(image + img_mean).astype(np.uint8)

# 获取原始图像
ori_img = get_original_image(image[0], imagemeta[0], img_mean)

# 展示原图像和送⼊⽹络中图像
rgd_image = np.round(image + img_mean).astype(np.uint8)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
axes[0].imshow(ori_img.astype('uint8'))
axes[0].set_title("原图像")
axes[1].imshow(rgd_image[0])
axes[1].set_title("送 ⼊ 网 络中的图像")
plt.show()

# 原始图像为375，500，3
# 送入网络为1，1216，1216，3
print(ori_img.shape, image.shape)

# 预训练
# coco数据集的class，共80个类别：人，自行车，火车，。。。
classes = ['bg', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
           'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
           'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
           'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# 实例化模型
model = faster_rcnn.FasterRCNN(num_classes=len(classes))
model((image, imagemeta, bbox, label), training=True)
model.load_weights('./weight/faster_rcnn1.h5')
model.summary()

# RPN获取候选区域：输⼊图像和对应的元信息，输出是候选的位置信息
proposals = model.simple_test_rpn(image[0], imagemeta[0])
# 绘制
# 大概1513个，完全看不清

# visualize.draw_boxes(rgd_image[0],boxes=proposals[:,:4]*1216)
# 选10个就行
visualize.draw_boxes(rgd_image[0], boxes=proposals[11:20, :4] * 1216)

plt.show()

# FastRcnn检测

res = model.simple_test_bboxes(image[0], imagemeta[0], proposals)
# res是⼀个字典，其结果如下所示：rois是⽬标框，class_ids是所属的类 别，scores是置信度。
print(res)

visualize.display_instances(ori_img,res['rois'],res["class_ids"],classes,res["scores"])
plt.show()
