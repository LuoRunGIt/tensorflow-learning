import tensorflow as tf
# 设置真实值和预测值
y_true = [[0.], [0.]]
y_pred = [[1.], [1.]]
# 实例化MAE损失
mae = tf.keras.losses.MeanAbsoluteError()
# 计算损失结果
m=mae(y_true, y_pred).numpy()
#以绝对误差为距离
print(m)