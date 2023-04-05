# 导⼊相应的包
import tensorflow as tf
# 设置真实值和预测值
y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
# 实例化交叉熵损失
cce = tf.keras.losses.CategoricalCrossentropy()
# 计算损失结果
k=cce(y_true, y_pred).numpy()
print(k)


"=====================二分类问题==========="
# 设置真实值和预测值
y_true = [[0], [1]]
y_pred = [[0.4], [0.6]]
# 实例化⼆分类交叉熵损失
bce = tf.keras.losses.BinaryCrossentropy()
# 计算损失结果
j=bce(y_true, y_pred).numpy()
print(j)