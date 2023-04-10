#图像增强
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

cat=plt.imread("./cat.jpg")
plt.imshow(cat)
plt.show()

cat1=tf.image.adjust_brightness(cat,0.5)
plt.imshow(cat1)
plt.show()

# 图像的色调通道的偏移量
cat2=tf.image.adjust_hue(cat,0.5)
plt.imshow(cat2)
plt.show()
