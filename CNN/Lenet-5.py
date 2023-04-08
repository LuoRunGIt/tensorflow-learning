import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
# 手写数字识别用cnn实现
# Lenet-5是89年提出的早期cnn
# 获取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, type(x_test), y_train.shape, y_test.shape)

# 这里需要先转np为tensor
# x_train = np.array(x_train, dtype=np.float32)
# x_test = np.array(x_test, dtype=np.float32)
# X_test = tf.convert_to_tensor(x_test)
# X_train = tf.convert_to_tensor(x_train)
# 这几句有点多余，错误的原因是少了个shape

# 数据处理，需要把数据处理加一个通道维数
train_images = tf.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
print(train_images.shape)
test_images = tf.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
print(test_images.shape)

# 模型搭建
net = keras.Sequential([
    # 相当于有6个5*5的卷积核,激活是sigmod
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', input_shape=(28, 28, 1)),
    # 最大池化层 池化层窗口大小，步长
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    #
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid', ),
    #
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 维度调整为一维
    # flatten表示降维默认为0，实际上可以有多种情况
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
], name="LeNet-5")
net.build(input_shape=(1, 28, 28, 1))
net.summary()  # 注意需要先编译模型，才能打印模型

# 定义优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.9)

# 定义损失函数、优化器和评价指标
# sparse_categorical_crossentropy交叉熵损失函数不需要one-hot编码
net.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy,
            metrics=keras.metrics.categorical_accuracy)

#模型训练
#validation_split: 0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。
net.fit(train_images,y_train,epochs=5,validation_split=0.1)
print("输入",train_images.shape,type(train_images))
print("测试",type(y_train))
#模型评估
score = net.evaluate(test_images, y_test, verbose=1)
print('Test accuracy:', score[1])
k=test_images[3000]
k=tf.reshape(k,(1,28,28,1))
print(k.shape)
p=net.predict(k)
print(p)
b = np.argmax(p)
print("预测值", b)
plt.figure()
plt.imshow(x_test[3000], cmap="gray")
plt.show()