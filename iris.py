# 绘图
import seaborn as sns
# 数值计算
import numpy as np
# sklearn中的相关⼯具
import matplotlib.pyplot as plt  # seaborn
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
# 逻辑回归
from sklearn.linear_model import LogisticRegressionCV
# tf.keras中使⽤的相关⼯具
# ⽤于模型搭建
import tensorflow as tf
from tensorflow.keras.models import Sequential
# 构建模型的层和激活⽅法
from tensorflow.keras.layers import Dense, Activation
# 数据处理的辅助⼯具
from tensorflow.keras import utils

# 读取数据
iris = sns.load_dataset(name="iris", data_home="C:\\Users\\骆润\\Desktop\\seaborn-data")
# 展示数据的前五⾏
print(iris.head())
pairplot_fig = sns.pairplot(iris, hue='species')
pairplot_fig.savefig("./iris1.png", dpi=400)
# seaborn 本身不能作图，用matplotlib
plt.show()

# # 花瓣和花萼的数据
X = iris.values[:, :4]
# 标签值
y = iris.values[:, 4]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)
print(train_y)
# 实例化分类器
lr = LogisticRegressionCV(max_iter=1000)  # 注意逻辑回归，需要设置迭代次数或阈值
# 训练
lr.fit(train_X, train_y)
print("Accuracy = {:.2f}".format(lr.score(test_X, test_y)))


# tf.keras实现
# 进⾏热编码
def one_hot_encode_object_array(arr):
    # 去重获取全部的类别
    uniques, ids = np.unique(arr, return_inverse=True)
    # 返回热编码的结果
    return utils.to_categorical(ids, len(uniques))


# 训练集热编码
train_y_ohe = one_hot_encode_object_array(train_y)
# 测试集热编码
test_y_ohe = one_hot_encode_object_array(test_y)
print(test_y_ohe)
# 其实我一直有个问题，为啥喜欢使用onehot编码而不直接用123这种简单的数来分类

# 利⽤sequential⽅式构建模型
model = Sequential([
    # 隐藏层1，激活函数是relu,输⼊⼤⼩有input_shape指定
    Dense(10, activation="relu", input_shape=(4,)),
    # 隐藏层2，激活函数是relu
    Dense(10, activation="relu"),
    # 输出层
    Dense(3, activation="softmax")  # 和种类对应
])
# 查看模型结构
model.summary()
# utils.plot_model(model,show_shapes=True)
# plt.show() #要装pydot

# optimizer 优化器
# loss 损失函数
# metrics 评价指标
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
print(type(train_X), type(test_X))

# 这里有2个坑，一个是模型输入必须是张量，另一个是张量与ndarray转换的时候，ndarray默认是int64 但张量是float32，所以需要做2次转换
train_X1 = np.array(train_X, dtype=np.float32)
train_y1 = np.array(train_y_ohe, dtype=np.float32)
test_X2 = np.array(test_X, dtype=np.float32)
test_y2 = np.array(test_y_ohe, dtype=np.float32)
# 数据类型转换
train_x = tf.convert_to_tensor(train_X1)
train_y = tf.convert_to_tensor(train_y_ohe)

test_x = tf.convert_to_tensor(test_X2)
test_y = tf.convert_to_tensor(test_y2)
# test_x = np.array(test_X, dtype=np.float)
# epochs,训练样本送⼊到⽹络中的次数，batch_size:每次训练批量
# verbose是日志显示，有三个参数可选择，分别为0,1和2。
# 当verbose=0时，简单说就是不输出日志信息 ，进度条、loss、acc这些都不输出。
# 当verbose=1时，带进度条的输出日志信息
model.fit(train_x, train_y, epochs=10, batch_size=1, verbose=1)

loss, accuracy = model.evaluate(test_x, test_y, verbose=1)
print("Accuracy = {:.2f}".format(accuracy))
