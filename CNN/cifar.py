import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100

# 加载Cifar10数据集
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# 加载Cifar100数据集
# (train_images, train_labels), (test_images, test_labels)=cifar100.load_data

print(train_images.shape, train_labels.shape)
plt.figure(figsize=(3,3))
plt.imshow(train_images[4])
plt.show()