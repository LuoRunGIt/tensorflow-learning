import numpy as np
import tensorflow as tf

from detection.core.bbox import transforms
from detection.utils.misc import *

# RPN网络的输出结果
class RPNTestMixin:
    
    def simple_test_rpn(self, img, img_meta):
        # 待检测的图像
        imgs = tf.Variable(np.expand_dims(img, 0))
        # 图像的元信息
        img_metas = tf.Variable(np.expand_dims(img_meta, 0))
        # 图像送入到backbone和fpn中进行特征提取与融合
        x = self.backbone(imgs, training=False)
        x = self.neck(x, training=False)
        # RPN网络输出的结果
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(x, training=False)
        # 获取候选区域
        proposals_list = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas, with_probs=False)
        return proposals_list[0]

# RCNN进行目标检测
class BBoxTestMixin(object):
    
    def _unmold_detections(self, detections_list, img_metas):
        return [
            self._unmold_single_detection(detections_list[i], img_metas[i])
            for i in range(img_metas.shape[0])
        ]
    # 对网络输出结果进行处理
    def _unmold_single_detection(self, detections, img_meta):
        zero_ix = tf.where(tf.not_equal(detections[:, 4], 0))
        detections = tf.gather_nd(detections, zero_ix)

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:, :4]
        class_ids = tf.cast(detections[:, 4], tf.int32)
        scores = detections[:, 5]

        boxes = transforms.bbox_mapping_back(boxes, img_meta)

        return {'rois': boxes.numpy(),
                'class_ids': class_ids.numpy(),
                'scores': scores.numpy()}

    def simple_test_bboxes(self, img, img_meta, proposals):
        # 待检测的图像
        imgs = tf.Variable(np.expand_dims(img, 0))
        # 图像元信息
        img_metas = tf.Variable(np.expand_dims(img_meta, 0))
        # RPN网络输出的候选区域
        rois_list = [tf.Variable(proposals)]
        # 图像送入网络中进行特征提取与融合
        x = self.backbone(imgs, training=False)
        P2, P3, P4, P5, _ = self.neck(x, training=False)
        # 获取RCNN中需要的特征图
        rcnn_feature_maps = [P2, P3, P4, P5]
        
        # 根据RPN网络获取的候选区域进行ROIPooling
        pooled_regions_list = self.roi_align(
            (rois_list, rcnn_feature_maps, img_metas), training=False)
        # 进行目标检测，获取检测结果
        rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list = \
            self.bbox_head(pooled_regions_list, training=False)
        # 获取原图上的检测结果
        detections_list = self.bbox_head.get_bboxes(
            rcnn_probs_list, rcnn_deltas_list, rois_list, img_metas)
        # 返回结果
        return self._unmold_detections(detections_list, img_metas)[0]