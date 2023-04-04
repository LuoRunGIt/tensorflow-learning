# 导入工具包
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
# 定义一个model的子类
class MyModel(keras.Model):

    #super()用来调用父类(基类)的方法，__init__()是类的构造方法，
#super().__init__() 就是调用父类的init方法， 同样可以使用super()去调用父类的其他方法

    # 定义网络的层结构
    def __init__(self):
        super(MyModel,self).__init__()
        # 第一层隐层
        self.layer1 = layers.Dense(3,activation="relu",name="layer1")
        # 第二个隐层
        self.layer2 = layers.Dense(2,activation="relu",name="layer2")
        # 输出层
        self.layer3 = layers.Dense(2,activation="sigmoid",name = "layer3")
    # 定义网络的前向传播
    def call(self,inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        outputs = self.layer3(x)
        return outputs

model = MyModel()
# 设置输入,这个表示输入层
x = tf.ones((1,4))
print(x)
y = model(x)
model.summary()