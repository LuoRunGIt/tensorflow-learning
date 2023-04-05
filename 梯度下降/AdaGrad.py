# 导⼊相应的⼯具包
import tensorflow as tf

# 实例化优化⽅法：SGD
opt = tf.keras.optimizers.Adagrad(
    learning_rate=0.1, initial_accumulator_value=0.1, epsilon=1e-07)
# 定义要调整的参数
var = tf.Variable(1.0)


# 定义损失函数：⽆参但有返回值
def loss(): return (var ** 2) / 2.0


# 计算梯度，并对参数进⾏更新，
opt.minimize(loss, [var]).numpy()
# 展示参数更新结果
var.numpy()
