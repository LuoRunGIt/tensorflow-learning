# 导⼊相应的⼯具包
import tensorflow as tf

# 实例化优化⽅法RMSprop
# RMSProp（Root Mean Square Prop）算法将这些梯度按元 素平⽅做指数加权移动平均
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
# 定义要调整的参数
var = tf.Variable(1.0)


# 定义损失函数：⽆参但有返回值
def loss(): return (var ** 2) / 2.0


# 计算梯度，并对参数进⾏更新，
opt.minimize(loss, [var]).numpy()
# 展示参数更新结果
k = var.numpy()
print(k)
