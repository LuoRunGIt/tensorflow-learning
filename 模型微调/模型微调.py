import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

traindir = 'E:\\BaiduNetdiskDownload\\课程资料\\03-深度学习与CV\\02.代码\\03.图像分类\\hotdog\\train'
testdir = 'E:\\BaiduNetdiskDownload\\课程资料\\03-深度学习与CV\\02.代码\\03.图像分类\\hotdog\\test'
image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.)

# 获取训练集数据
train_data_gen = image_gen.flow_from_directory(traindir, batch_size=32, target_size=(224, 224), shuffle=True)
# 获取测试集数据
test_data_gen = image_gen.flow_from_directory(testdir, batch_size=32, target_size=(224, 224), shuffle=True)
image, label = next(train_data_gen)
plt.figure(figsize=(10, 10))
for n in range(15):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(image[n])
    plt.axis('off')
plt.show()

# 读入已有模型
ResNet50 = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3))
for layer in ResNet50.layers:
    layer.trainable = False
net = tf.keras.models.Sequential()
net.add(ResNet50)  # 模型迁移
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(2, activation="sigmoid"))  # 修改输出

# 模型编译
net.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=["accuracy"])
#validation_data指定验证集 validation_split=0.2自动切分
net.fit(train_data_gen, steps_per_epoch=10, epochs=3, batch_size=32, validation_data=test_data_gen, validation_steps=10)

score = net.evaluate(test_data_gen, verbose=1)
print(score,type(score))
