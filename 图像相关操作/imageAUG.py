#图像增强
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

cat=plt.imread("./cat.jpg")
plt.imshow(cat)
plt.show()

#翻转或裁剪
#左右翻转这个random表示的是随机
cat1=tf.image.random_flip_left_right(cat)
plt.imshow(cat1)
plt.show()

cat1_1=tf.image.flip_left_right(cat)
plt.imshow(cat1_1)
plt.show()

cat2_1=tf.image.flip_up_down(cat)
plt.imshow(cat2_1)
plt.show()

#上下翻转
cat2=tf.image.random_flip_up_down(cat)
plt.imshow(cat2)
plt.show()

cat3=tf.image.random_crop(cat,(200,200,3))
plt.imshow(cat3)
plt.show()