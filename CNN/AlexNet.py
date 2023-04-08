import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

net = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation="relu"),
    # layer本身就是层的意思
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    # 设成same会自动加padding，保证输出是同样的大小。
    # padding = same，是指输出图像要和输入一样，至于padding是多少，是推断出来的。
    tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    # 一维展开
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation="relu"),
    # 随机失活
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4096, activation="relu"),
    # 随机失活
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")

], name='my_Alexnet')
# 这里我其实不知清楚输入的维度，227x227应该是原始在 imagenet比赛时图片的维度
# 这里因为数据集使用的是手写数字识别，所以需要改变维度
net.build(input_shape=(1, 227, 227, 1))
net.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, type(x_test), y_train.shape, y_test.shape)

# 数据处理，需要把数据处理加一个通道维数
# train_images = tf.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
# test_images = tf.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
# print(train_images.shape, test_images.shape)
# 这一步很关键，他把一维数组处理成了图片
train_images = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
test_images = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))


# 对训练数据进行抽样
def get_train(size):
    # 随机生成index
    index = np.random.randint(0, train_images.shape[0], size)
    # 选择图像并进行resize
    # image.resize_with_pad这个函数可以调整图像，并且可以使用如非线性插值等方法对图像进行处理
    resized_image = tf.image.resize_with_pad(train_images[index], 227, 227)
    return resized_image.numpy(), y_train[index]


# 对测试数据进行抽样
def get_test(size):
    # 随机生成index
    index = np.random.randint(0, test_images.shape[0], size)
    print("index:", index)
    # 选择图像并进行resize
    resized_image = tf.image.resize_with_pad(test_images[index], 227, 227)
    return resized_image.numpy(), y_test[index]


# 这里进行了随机采样，并且维度实际上从28x28拉伸了
train_images, train_label = get_train(10000)
test_images, test_labels = get_test(128)
print(train_label[4])
print(type(train_images))
plt.imshow(train_images[4].astype(np.int8).squeeze(), cmap='gray')
plt.show()
#momentum表示动量，实际上我这次实验没有使用到动量
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.0,nesterov=False)
# 优化器损失函数评价指标
# 'accuracy'表示预测值为int 真实值也为int
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy'])
# validation_split: 0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。
# verbose表示日志参数数量，即跑的时候那个进度条
net.fit(train_images, train_label, batch_size=128, epochs=3, validation_split=0.1,
        verbose=1)

# 模型评估
score = net.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy:', score[1])
print(test_images.shape, type(test_images), train_label.shape)
k = train_images[4]
k = k[np.newaxis, :, :, :]
print(k.shape, type(k))
p = net.predict(k)
print(p)
b = np.argmax(p)
print("预测值", b)
# 这里我训练的时候出现了一个情况，就是没有预测值都是nan，说明模型训练计算不出结果
# 排查的时候我发现loss在第2次就变为NaN了，因此我降低了学习率，同时扩大了训练集的数量
# 不过最后预测错了
plt.figure()
plt.imshow(train_images[4].astype(np.int8).squeeze(), cmap='gray')
plt.show()
