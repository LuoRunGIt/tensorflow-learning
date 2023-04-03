import tensorflow as tf
import numpy as np

# 张量
# 创建int32类型的0维张量，即标量
rank_0_tensor = tf.constant(4)  # 所以张量默认是int32
print(rank_0_tensor)
# 创建float32类型的1维张量
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
# 创建float16类型的⼆维张量
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)

rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]], ])
print(rank_3_tensor)

# 张量和numpy之间的转换
k = np.array(rank_2_tensor)
print(k)
k1 = rank_2_tensor.numpy()
print(k1)

# 运算
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]])
print(tf.add(a, b), "\n")  # 计算张量的和
print(tf.multiply(a, b), "\n")  # 计算张量的元素乘法，对应元素相乘
print(tf.matmul(a, b), "\n")  # 计算乘法，点乘

print("\n求和 ", tf.reduce_sum(a))  # 求和,仅仅只针对这个张,返回值仍是张量
print("\n均值", tf.reduce_mean(a))  # 平均值
print("\n最大值", tf.reduce_max(a))  # 最⼤值
print("\n最小值", tf.reduce_min(a))  # 最⼩值
print("\n最大值索引", tf.argmax(a))  # 最⼤值的索引
print("\n最小值索引", tf.argmin(a))  # 最⼩值的索引
k2 = tf.reduce_sum(a)
k3 = tf.argmin(a)
print(type(k2), type(k3))

# 张量中的变量
my_variable = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy)
