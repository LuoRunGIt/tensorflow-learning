import tensorflow as tf

from detection.models.backbones import resnet
from detection.models.necks import fpn
from detection.models.rpn_heads import rpn_head
from detection.models.bbox_heads import bbox_head
from detection.models.roi_extractors import roi_align
from detection.models.detectors.test_mixins import RPNTestMixin, BBoxTestMixin

from detection.core.bbox import bbox_target


# 模型构建类
class FasterRCNN(tf.keras.Model, RPNTestMixin, BBoxTestMixin):
    # 初始化
    def __init__(self, num_classes, **kwags):
        super(FasterRCNN, self).__init__(**kwags)
        # 类别个数
        self.NUM_CLASSES = num_classes
        
        # RPN 部分
        # Anchor参数
        self.ANCHOR_SCALES = (32, 64, 128, 256, 512)
        self.ANCHOR_RATIOS = (0.5, 1, 2)
        self.ANCHOR_FEATURE_STRIDES = (4, 8, 16, 32, 64)
        
        # Bounding box的均值和标准差
        self.RPN_TARGET_MEANS = (0., 0., 0., 0.)
        self.RPN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)
        
        # RPN 训练参数配置
        self.PRN_BATCH_SIZE = 256
        self.RPN_POS_FRAC = 0.5
        self.RPN_POS_IOU_THR = 0.7
        self.RPN_NEG_IOU_THR = 0.3

        # 候选区域参数设置
        self.PRN_PROPOSAL_COUNT = 2000
        self.PRN_NMS_THRESHOLD = 0.7
        
        # RCNN 部分
        # Bounding box的均值和方差
        self.RCNN_TARGET_MEANS = (0., 0., 0., 0.)
        self.RCNN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)
        
        # ROIpooling后特征图的大小
        self.POOL_SIZE = (7, 7)
        
        # RCNN 训练参数
        self.RCNN_BATCH_SIZE = 256
        self.RCNN_POS_FRAC = 0.25
        self.RCNN_POS_IOU_THR = 0.5
        self.RCNN_NEG_IOU_THR = 0.5
        
        # Boxes保留的参数配置
        self.RCNN_MIN_CONFIDENCE = 0.7
        self.RCNN_NME_THRESHOLD = 0.3
        self.RCNN_MAX_INSTANCES = 100
        
        # RCNN目标检测的目标值设置
        self.bbox_target = bbox_target.ProposalTarget(
            target_means=self.RCNN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS, 
            num_rcnn_deltas=self.RCNN_BATCH_SIZE,
            positive_fraction=self.RCNN_POS_FRAC,
            pos_iou_thr=self.RCNN_POS_IOU_THR,
            neg_iou_thr=self.RCNN_NEG_IOU_THR)
                
        # backbone:resnet
        self.backbone = resnet.ResNet(
            depth=101, 
            name='res_net')
        # fpn进行特征融合
        self.neck = fpn.FPN(
            name='fpn')
        # RPN网络生成候选区域
        self.rpn_head = rpn_head.RPNHead(
            anchor_scales=self.ANCHOR_SCALES,
            anchor_ratios=self.ANCHOR_RATIOS,
            anchor_feature_strides=self.ANCHOR_FEATURE_STRIDES,
            proposal_count=self.PRN_PROPOSAL_COUNT,
            nms_threshold=self.PRN_NMS_THRESHOLD,
            target_means=self.RPN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            num_rpn_deltas=self.PRN_BATCH_SIZE,
            positive_fraction=self.RPN_POS_FRAC,
            pos_iou_thr=self.RPN_POS_IOU_THR,
            neg_iou_thr=self.RPN_NEG_IOU_THR,
            name='rpn_head')
        # ROIpooling
        self.roi_align = roi_align.PyramidROIAlign(
            pool_shape=self.POOL_SIZE,
            name='pyramid_roi_align')
        # 目标检测部分
        self.bbox_head = bbox_head.BBoxHead(
            num_classes=self.NUM_CLASSES,
            pool_size=self.POOL_SIZE,
            target_means=self.RCNN_TARGET_MEANS,
            target_stds=self.RCNN_TARGET_STDS,
            min_confidence=self.RCNN_MIN_CONFIDENCE,
            nms_threshold=self.RCNN_NME_THRESHOLD,
            max_instances=self.RCNN_MAX_INSTANCES,
            name='b_box_head')
    # 前向传播
    def call(self, inputs, training=True):
        """

        :param inputs: [1, 1216, 1216, 3], [1, 11], [1, 14, 4], [1, 14]
        :param training:
        :return:
        """
        # 判断是训练或预测阶段
        if training: # training
            imgs, img_metas, gt_boxes, gt_class_ids = inputs
        else: # inference
            imgs, img_metas = inputs
        # backbone
        # [1, 304, 304, 256] => [1, 152, 152, 512]=>[1,76,76,1024]=>[1,38,38,2048]
        C2, C3, C4, C5 = self.backbone(imgs, 
                                       training=training)
        # fpn特征融合
        # [1, 304, 304, 256] <= [1, 152, 152, 256]<=[1,76,76,256]<=[1,38,38,256]=>[1,19,19,256]
        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5], 
                                       training=training)
        # 融合后的特征分别送入到rpn和rcnn网络中
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]
        # RPN网络的输出结果
        # [1, 369303, 2] [1, 369303, 2], [1, 369303, 4], includes all anchors on pyramid level of features
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(
            rpn_feature_maps, training=training)
        # proposal layer的输出结果
        # [369303, 4] => [215169, 4], valid => [6000, 4], performance =>[2000, 4],  NMS
        proposals_list = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas)
        # 获取生成的候选区域
        if training: # get target value for these proposal target label and target delta
            rois_list, rcnn_target_matchs_list, rcnn_target_deltas_list = \
                self.bbox_target.build_targets(
                    proposals_list, gt_boxes, gt_class_ids, img_metas)
        else:
            rois_list = proposals_list
        # roipooling
        # rois_list only contains coordinates, rcnn_feature_maps save the 5 features data=>[192,7,7,256]
        pooled_regions_list = self.roi_align(#
            (rois_list, rcnn_feature_maps, img_metas), training=training)
        # 模型检测结果
        # [192, 81], [192, 81], [192, 81, 4]
        rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list = \
            self.bbox_head(pooled_regions_list, training=training)
        if training:
            # 训练阶段计算损失
            rpn_class_loss, rpn_bbox_loss = self.rpn_head.loss(
                rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas)
            
            rcnn_class_loss, rcnn_bbox_loss = self.bbox_head.loss(
                rcnn_class_logits_list, rcnn_deltas_list, 
                rcnn_target_matchs_list, rcnn_target_deltas_list)
            # 返回损失
            return [rpn_class_loss, rpn_bbox_loss, 
                    rcnn_class_loss, rcnn_bbox_loss]
        else:
            # 预测阶段获取检测结果
            detections_list = self.bbox_head.get_bboxes(
                rcnn_probs_list, rcnn_deltas_list, rois_list, img_metas)
            # 返回预测结果
            return detections_list
