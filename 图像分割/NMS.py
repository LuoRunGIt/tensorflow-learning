# 非极大值抑制
import numpy as np

bounding = [(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304),(1,1,1,1),(2,2,2,2)]
confidence_score = [0.9, 0.65, 0.8,0.85,0.7]
threshold = 0.3


def nms(bboxes, confidence, threshold=0.3):
    '''

    :param bboxes:同类别候选框坐标
    :param confidence: 同类别候选框分数
    :param threshold: IOU阈值，当IOU阈值大于threshold时直接去除
    :return: 返回值是一个box列表和对应的分数列表 【】 【】
    '''
    # 传入为空直接返回空
    if len(bboxes) == 0:
        return [], []

    # 将数组转为numpy
    bboxes = np.array(bboxes)
    score = np.array(confidence)
    # print(bboxes)

    x1 = bboxes[:, 0]  # 取了所有左上角坐标
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    # print(x1)

    picked_box = []  # 返回框
    picked_score = []  # 返回值

    # 排序
    order = np.argsort(score)  # 这里排序的结果不是置信度，而是从小到大的下标[1，2，0]
    areas = (x2 - x1) * (y2 - y1)
    print(order)
    while order.size > 0:
        # 先把最大的加入返回结果
        index = order[-1]

        picked_box.append(bboxes[index])
        picked_score.append((confidence[index]))

        # 随后做一个IOU操作
        x11 = np.maximum(x1[index], x1[order[:-1]])
        print("x11",x11)  # 注意这里时批量计算，算出的结果是矩阵
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])

        w = np.maximum(0.0, x22 - x11)
        h = np.maximum(0.0, y22 - y11)
        inter = w * h
        print("面积矩阵",inter)
        ratio = inter /np.maximum(1.0 ,(areas[index] + areas[order[:-1]] - inter))
        print(ratio.shape)
        keep = np.where(ratio < threshold)

        print("keep",keep[0])
        order = order[keep]
        print("order",order)

    return picked_box, picked_score


boxes,score=nms(bounding, confidence_score, threshold=threshold)
print(boxes,score)
