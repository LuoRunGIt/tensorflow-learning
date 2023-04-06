# 导⼊相应的⼯具包
import keras.losses
import tensorflow as tf
from tensorflow.keras import regularizers

# 创建模型
model = tf.keras.models.Sequential()
# L2正则化，lambda为0.01
model.add(tf.keras.layers.Dense(16, kernel_regularizer=regularizers.l1(0.001),
                                activation='relu', input_shape=(10,)))
# L1正则化，lambda为0.01
model.add(tf.keras.layers.Dense(16, kernel_regularizer=regularizers.l2(0.002),
                                activation='relu'))
# L1L2正则化，lambda1为0.01,lambda2为0.01
model.add(tf.keras.layers.Dense(16, kernel_regularizer=regularizers.L1L2(0.001, 0.002),
                                activation='relu'))
