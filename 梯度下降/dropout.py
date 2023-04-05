# 导入工具包
import tensorflow as tf
import numpy as np

# 随机失活
# 定义dropout层
# 定义dropout层,每⼀个神经元有0.2的概率被失活，未被失活的输⼊将按1/1-rate 0.2放大
# Dropout也可用于可见层，如神经网络的输入。在这种情况下，就要把Dropout层作为网络的第一层，并将input_shape参数添加到层中，来制定预期输入。
# https://blog.csdn.net/yangwohenmai1/article/details/123346240
layer = tf.keras.layers.Dropout(0.2, input_shape=(4,))
# 定义输入数据
data = np.arange(1, 11).reshape(5, 2).astype(np.float32)
print(data)
# 对输入数据进行随机失活
# train 为false返回原始值
outputs1 = layer(data, training=False)
outputs = layer(data, training=True)
print(outputs1, outputs)
