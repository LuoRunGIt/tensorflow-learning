from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 数据处理的辅助⼯具
from tensorflow.keras import utils
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers  # 正则化

# 获取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, type(x_test), y_train.shape, y_test.shape)
# 注意此时训练集和测试集数据均为np ndarray不是tensor
# 数据集需要转换为tensor

# 查看数据集
plt.figure()
plt.imshow(x_test[1], cmap="gray")
plt.show()

# 数据处理
# 把数据从28*28 改为 784
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 数据归一化，这里只需要对训练集归一化
x_train = x_train / 255
x_test = x_test / 255


def one_hot_encode_object_array(arr):
    # 去重获取全部的类别
    uniques, ids = np.unique(arr, return_inverse=True)
    # 返回热编码的结果
    return utils.to_categorical(ids, len(uniques))


# 以onehot编码的形式处理输出y
# 训练集热编码
train_y_ohe = one_hot_encode_object_array(y_train)
# 测试集热编码
test_y_ohe = one_hot_encode_object_array(y_test)

# 模型搭建
model = keras.Sequential([
    # 第一个隐藏层
    layers.Dense(512, activation="relu", kernel_initializer="he_normal",
                 name="layer1", input_shape=(784,)),
    #
    layers.BatchNormalization(),
    # 设置一个dropout
    layers.Dropout(0.2),
    # 第二个隐藏层
    layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # 输出层
    layers.Dense(10, activation="softmax")
],
    name="my_minist")
model.summary()
# 模型训练
# 模型编译，损失函数，优化器（即更新梯度），评价指标
# 多分类交叉熵损失函数
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
              metrics=tf.keras.metrics.categorical_accuracy)
# 训练输入必须是tensor
x_train = np.array(x_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)

X_train = tf.convert_to_tensor(x_train)
Y_train = tf.convert_to_tensor(train_y_ohe)
X_test = tf.convert_to_tensor(x_test)
Y_test = tf.convert_to_tensor(test_y_ohe)
history = model.fit(X_train, Y_train, epochs=4, batch_size=128, validation_data=(X_test, Y_test), verbose=1)
# Shapes (None, 1) and (None, 10) are incompatible,这里导入的时候没注意是one-hot编码
# 损失函数
# history里保存了记录
plt.figure()
plt.plot(history.history['loss'], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.grid()
plt.show()
# 准确率
plt.figure()
print(history.history)
# 训练集准确率
plt.plot(history.history['categorical_accuracy'], label="train")
plt.plot(history.history['val_categorical_accuracy'], label="val")
plt.legend()
plt.grid()
plt.show()
'''
# 回调函数(这里回调的效果是保存日志)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./graph")
# 训练
history = model.fit(X_train, Y_train, epochs=4, validation_data=(X_test, Y_test), batch_size=128,
                    verbose=1, callbacks=[tensorboard])
'''
k = model.evaluate(X_test, Y_test, verbose=1)
print(k, type(k))

model.save("./model.h5")
'''
loadmodel = tf.keras.models.load_model("model.h5")
loadmodel.evaluate(x_test,y_test,verbose=1)
'''
# 注意这里转化的维度（10000，784）
print(X_test.shape)
# 这里是[x.test[1]] （1，784）维度一致
x_test1 = tf.convert_to_tensor([x_test[1]])
print(x_test1.shape)
p = model.predict(x_test1)
# 判断输出值
print(p, type(p), p.shape)
# 最大值下标
b = np.argmax(p)
print("预测值", b)
