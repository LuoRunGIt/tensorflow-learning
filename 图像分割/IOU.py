import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 定义方法计算IOU
def IOU(box1, box2, wh=False):
    # 判断表示方式
    if wh == False:
        # 极坐标表示
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        # 中心点坐标表示
        # 第一框
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        # 第二框
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)
    # 获取交集的左上角和右下角坐标
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax2, xmax1])
    yy2 = np.min([ymax1, ymax2])
    # 计算交集面积
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算并的面积，面积相加-交的面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    uion_area = area1 + area2 - inter_area
    # IOU
    IOU = inter_area / (uion_area + 1e-6)
    return IOU


# 真实框 预测框
true_box = [100, 35, 398, 400]
pre_box = [40, 150, 355, 398]

# 将框绘制在图像上
img = plt.imread("./dog.jpeg")
fig = plt.imshow(img)
# 将真实框和预测框绘制在图像上
fig.axes.add_patch(
    plt.Rectangle((true_box[0], true_box[1]), width=true_box[2] - true_box[0], height=true_box[3] - true_box[1],
                  fill=False, edgecolor="blue", linewidth=2))
fig.axes.add_patch(
    plt.Rectangle((pre_box[0], pre_box[1]), width=pre_box[2] - pre_box[0], height=pre_box[3] - pre_box[1], fill=False,
                  edgecolor="red", linewidth=2))
plt.show()
print(IOU(true_box, pre_box))
# 一般来说大于0.5是可接受的
