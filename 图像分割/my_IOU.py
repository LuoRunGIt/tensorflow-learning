import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


# 定义方法计算IOU
# IOU下的坐标，x轴朝下，y轴是朝右
# 极坐标标表示下一个图形只要关注左上角和右下角即（x1,y1）(x2,y2)
# 另一个矩形的左上角和右下角则表示为（a1,b1）(a2,b2)
#
# 我们先定义函数的输入

def IOU(box1, box2, wh=False):
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        # 中心坐标表示下需要进行一次转换
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        # 第二框
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    # 计算交集坐标，注意这里可能出现没有交集的情况
    xi1 = max(xmin1, xmin2)
    yi1 = max(ymin1, ymin2)

    xi2 = min(xmax2, xmax1)
    yi2 = min(ymax1, ymax2)

    # 计算交集的面积
    inter = (np.max([0, xi2 - xi1])) * (np.max([0, yi2 - yi1]))

    # 计算并的面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    uion = area2 + area1 - inter

    Iou = inter / (uion + 1e-6)
    return Iou


true_box = [100, 35, 398, 400]
pre_box = [40, 150, 355, 398]
iou = IOU(true_box, pre_box)
print(iou)

img = plt.imread("./dog.jpeg")
fig = plt.imshow(img)

# 设置坐标
fig.axes.add_patch(
    plt.Rectangle((true_box[0], true_box[1]), width=true_box[3] - true_box[1], height=true_box[2] - true_box[0],
                  fill=False, edgecolor="blue", linewidth=2
                  ))
fig.axes.add_patch(
    plt.Rectangle((pre_box[0], pre_box[1]), width=pre_box[2] - pre_box[0], height=pre_box[3] - pre_box[1],
                  fill=False, edgecolor="red", linewidth=2))
plt.show()
