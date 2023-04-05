# 此类损失函数常常用于目标检测
# 导⼊相应的包
import tensorflow as tf

# 设置真实值和预测值
y_true = [[0], [1]]
y_pred = [[0.6], [0.4]]
# 实例化smooth L1损失
h = tf.keras.losses.Huber()
# 计算损失结果
n = h(y_true, y_pred).numpy()
print(n)
