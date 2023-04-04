# 导入工具包
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
# 定义模型的输入，这个是函数式编程的方式
inputs = keras.Input(shape=(3,), name='input')
# 第一个隐层
x = layers.Dense(3, activation="relu", name="layer1")(inputs)
# 第二个隐层
x = layers.Dense(2, activation="relu", name="layer2")(x)
# 输出层
outputs = layers.Dense(2, activation="sigmoid", name="output")(x)
# 创建模型
model = keras.Model(inputs=inputs, outputs=outputs,
                    name="Functional API Model")
model.summary()