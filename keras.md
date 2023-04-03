import tensorflow as tf
from tensorflow import keras
# 配置优化⽅法，损失函数和评价指标
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
 loss='categorical_crossentropy',
 metrics=['accuracy'])

# 指明训练数据集，训练epoch,批次⼤⼩和验证集数据
model.fit/fit_generator(dataset, epochs=10,
 batch_size=3,
 validation_data=val_dataset,
 )

# 指明评估数据集和批次⼤⼩
model.evaluate(x, y, batch_size=32)

# 对新的样本进⾏预测
model.predict(x, batch_size=32)

# 只保存模型的权重
model.save_weights('./my_model')
# 加载模型的权重
model.load_weights('my_model')

# 保存模型架构与权重在h5⽂件中
model.save('my_model.h5')
# 加载模型：包括架构和对应的权重
model = keras.models.load_model('my_model.h5')