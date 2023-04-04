# 导⼊相应的⼯具包
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
# 数字中的score
x = tf.constant([0.2,0.02,0.15,1.3,0.5,0.06,1.1,0.05,3.75])
# 将其送⼊到softmax中计算分类结果
y = tf.nn.softmax(x)
# 将结果进⾏打印
print(y)
print(tf.reduce_sum(y))