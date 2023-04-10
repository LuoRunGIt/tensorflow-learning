import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 获取数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 将数据转换为4维的形式
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 左右翻转
datagen = tf.keras.preprocessing.image.ImageDataGenerator(shear_range=10, horizontal_flip=True)

for x, y in datagen.flow(x_train, y_train, batch_size=9):
    print(x.shape, y.shape)
    print(type(x), type(y))
    plt.figure(figsize=(8, 8))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x[i].reshape(28, 28), cmap='gray')
        plt.title(y[i])
    plt.show()
    break

# 颠倒
datagen1 = tf.keras.preprocessing.image.ImageDataGenerator(shear_range=10, vertical_flip=True)
cat = plt.imread("./cat.jpg")
cat = cat[np.newaxis, :, :, :]  # 增加一个维度
print(cat.shape)
f = datagen1.flow(cat)
# print(f.shape)
# 注意范围修改到
plt.imshow(f[0].reshape(218, 284, 3) / 255)
plt.show()
