# concat 网络，连接；连接两个字符串、函数
import tensorflow as tf

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
k1 = tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
k2 = tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
k3 = tf.concat([t1, t2], -1)
print(k1, k2,k3)

# axis=0,1,-1 表示按第0个维度，第一个维度或倒数第一个维度进行拼接
# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
# tf.shape(tf.concat([t3, t4], axis=0))  # [4, 3]
# tf.shape(tf.concat([t3, t4], axis=1))  # [2, 6]
