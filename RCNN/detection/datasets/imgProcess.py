import numpy as np
from detection.datasets import transforms, utils
import cv2

class ImgProcess():
    def __init__(self,image):
        # 要进行预测的图像
        self.data_path = image
        # 输入网络中的图像尺寸
        self.target_size = 1024
        self.scale = (1024, 1024)
        # 像素RGB的均值
        self.pixel_means = np.array([[[122.7717, 115.9465, 102.9801]]])
        self.img_transform = transforms.ImageTransform(self.scale, self.pixel_means, [1.,1.,1.], 'fixed')
        self.flip_ratio=0.5

    def imgProcess(self):
        # 读取图像
        image = self.image_read(self.data_path)
        # 获取原始图像的尺寸
        ori_shape = image.shape
        # 进行尺度调整后的图像及调整的尺度
        image, image_scale, image_shape = self.prep_im_for_blob(image, self.pixel_means, self.target_size)
        # 获取尺度变换后图像的尺寸
        pad_shape = image.shape
        # 图像的基本信息
        img_meta_dict = dict({
            'ori_shape': ori_shape,
            'img_shape': image_shape,
            'pad_shape': pad_shape,
            'scale_factor': image_scale
        })
        # 将字典转换为列表的形式
        image_meta = self.compose_image_meta(img_meta_dict)
        return image, image_meta

    def image_read(self, imname):
        # opencv 中默认图片色彩格式为BGR
        image = cv2.imread(imname)
        # 将图片转成RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        return image

    def prep_im_for_blob(self, im, pixel_means, target_size):
        "对输入的图像进行处理"
        im = im.astype(np.float32, copy=False)
        # 减去均值
        im -= pixel_means
        # 图像的大小
        im_shape = im.shape
        # 最短边
        im_size_min = np.min(im_shape[0:2])
        # 最长边
        im_size_max = np.max(im_shape[0:2])
        # 长边变换为1024
        im_scale = float(target_size) / float(im_size_max)
        # 根据变换比例对图像进行resize
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        shape = (target_size, target_size, im.shape[-1])
        pad = np.zeros(shape, dtype=im.dtype)
        pad[:im.shape[0], :im.shape[1], ...] = im
        # 返回im 和 im_scale
        return pad, im_scale, im.shape

    def compose_image_meta(self,img_meta_dict):
        '''将图像的原信息转换为列表.
        '''
        # 获取到字典中的信息
        ori_shape = img_meta_dict['ori_shape']
        img_shape = img_meta_dict['img_shape']
        pad_shape = img_meta_dict['pad_shape']
        scale_factor = img_meta_dict['scale_factor']
        flip = 0
        # 转换为一维列表
        img_meta = np.array(
            ori_shape +  # size=3
            img_shape +  # size=3
            pad_shape +  # size=3
            tuple([scale_factor]) + # size=1
            tuple([flip])  # size=1
        ).astype(np.float32)
        # 返回列表信息
        return img_meta


if __name__ == '__main__':
    import visualize
    imageProcess = ImgProcess('/Users/yaoxiaoying/Desktop/bjyx.jpeg')
    # print(pascal.gt_labels)
    img, image_meta = imageProcess.imgProcess()
    print(img.shape)
    print(image_meta.shape)
    # print (len(pascal.gt_labels))
    from detection.datasets.utils import get_original_image
    import matplotlib.pyplot as plt
    import cv2

    # img, img_meta, bboxes, labels = data[0].image
    img_mean = [[122.7717, 115.9465, 102.9801]]
    img_std = (1., 1., 1.)
    rgb_img = np.round(img + img_mean).astype('uint8')
    # ori_img = get_original_image(data['image'], img_meta, img_mean)

    # visualize.display_instances(rgb_img[0], bboxes[0], labels[0], pascal.classes)
    # plt.show()
    from detection.datasets import pascal_voc
    import matplotlib.pyplot as plt
    import numpy as np
    from detection.models.detectors import faster_rcnn
    import tensorflow as tf
    import visualize
    from detection.core.bbox import transforms

    pascal = pascal_voc.pascal_voc("train")
    image, imagemeta, bbox, label = pascal[8]
    # 图像的均值和标准差
    img_mean = (122.7717, 115.9465, 102.9801)
    img_std = (1., 1., 1.)
    # %%
    rgd_image = np.round(image + img_mean).astype(np.uint8)
    # coco数据集的class
    classes = ['bg', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    model = faster_rcnn.FasterRCNN(num_classes=len(classes))
    # %%
    # model((image, imagemeta, bbox, label), training=True)
    model((image, imagemeta), training=False)
    # %%
    # 加载训练好的weights
    model.load_weights("/Users/yaoxiaoying/Desktop/fasterRCNN/weights/faster_rcnn1.h5")
    # 加载训练好的weights
    # model = tf.keras.models.load_model("weights/faster_rcnn1.h5")
    # 获取候选区域
    proposals = model.simple_test_rpn(img, image_meta)
    # %%
    proposals
    # %%
    # rcnn进行预测
    res = model.simple_test_bboxes(img, image_meta, proposals)
    # %%
    res
    # %%
    # 获取原始图像
    from detection.datasets.utils import get_original_image

    # %%
    ori_img = get_original_image(img, image_meta, img_mean)
    # %%
    # 将检测结果绘制在图像上
    visualize.display_instances(ori_img, res['rois'], res['class_ids'], classes, res['scores'])
    plt.show()

