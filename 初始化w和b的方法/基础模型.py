# 导入工具包
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
#sequential方式 Sequential英文译为“顺序的”
# 定义model,构建模型
model = keras.Sequential([
    # 第一个隐层
    layers.Dense(3, activation="relu", kernel_initializer="he_normal",
                 name="layer1",input_shape=(3,)),
    # 第二个隐层
    layers.Dense(2, activation="relu",
                 kernel_initializer="he_normal", name="layer2"),
    # 输出层
    layers.Dense(2, activation="sigmoid",
                 kernel_initializer="he_normal", name="layer3")
    ],
    name="sequential"
)
model.summary()