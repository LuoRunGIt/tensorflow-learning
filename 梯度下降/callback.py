import tensorflow as tf
import numpy as np
# 定义回调函数
callback = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3)
# 定义一层的网络
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
# 模型编译
model.compile(tf.keras.optimizers.SGD(),loss='mse')
# 模型训练
history = model.fit(np.arange(100).reshape(5,20),np.array([0,1,0,1,0]),epochs=10,batch_size=1,verbose=1)
len(history.history['loss'])

#BN 批标准化
# 直接将其放⼊构建神经⽹络的结构中即可，
tf.keras.layers.BatchNormalization(
 epsilon=0.001, center=True, scale=True,
 beta_initializer='zeros', gamma_initializer='ones', )