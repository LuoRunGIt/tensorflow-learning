# 导⼊相应的⼯具包
import tensorflow as tf

# 实例化优化⽅法：SGD
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# 定义要调整的参数
k=var = tf.Variable(1.0)#var 可以变值但是不能变形状
print(k.shape,k)
# 定义损失函数：⽆参但有返回值
loss = lambda: (var ** 2) / 2.0
# 计算梯度，并对参数进⾏更新，步⻓为 `- learning_rate * grad`
opt.minimize(loss, [var]).numpy()
# 展示参数更新结果
k = var.numpy()
# 随机梯度下降
print(k)
# optimizers 优化器
# tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD",**kwargs)
# kwargs功能，它是一个字典如果输入为一个键值对，就会转化为字典
