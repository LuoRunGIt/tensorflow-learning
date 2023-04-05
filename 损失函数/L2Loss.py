import tensorflow as tf

# 设置真实值和预测值
y_true = [[0.], [1.]]
y_pred = [[1.], [1.]]
# 实例化MSE损失 平方和误差
mse = tf.keras.losses.MeanSquaredError()
# 计算损失结果
k = mse(y_true, y_pred).numpy()
print(k)
