import tensorflow as tf

#或称学习率衰减
# 对于前100000步，学习率为1.0，对于接下来的100000-110000步，学习率为0.5，之后的步骤学习率为0.1
# 分段常数衰减
boundaries = [100000, 110000]
# 不同的step对应的学习率
values = [1.0, 0.5, 0.1]
# 实例化进⾏学习的更新
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

# 指数衰减
# 这三个是官网案例
initial_learning_rate = 0.1
decay_steps = 100000
decay_rate = 0.96
tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=100000,
                                               decay_rate=0.96)


def decayed_learning_rate(step):
    return initial_learning_rate * decay_rate ^ (step / decay_steps)


# 1/t衰减
initial_learning_rate = 0.1
decay_steps = 1.0
decay_rate = 0.5
tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate, staircase=False,
                                               name=None)
