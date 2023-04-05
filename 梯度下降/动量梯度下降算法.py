# 导⼊相应的⼯具包
import tensorflow as tf
# 实例化优化⽅法：SGD 指定参数beta=0.9即momentum参数
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
# 定义要调整的参数，初始值
var = tf.Variable(1.0)
val0 = var.value()
# 定义损失函数
loss = lambda: (var ** 2)/2.0
#第⼀次更新：计算梯度，并对参数进⾏更新，步⻓为 `- learning_rate *
opt.minimize(loss, [var]).numpy()
val1 = var.value()
# 第⼆次更新：计算梯度，并对参数进⾏更新，因为加⼊了momentum,步⻓会增
opt.minimize(loss, [var]).numpy()#opt.minimize是梯度更新算法
val2 = var.value()
# 打印两次更新的步⻓
print("第⼀次更新步⻓={}".format((val0 - val1).numpy()))
print("第⼆次更新步⻓={}".format((val1 - val2).numpy()))