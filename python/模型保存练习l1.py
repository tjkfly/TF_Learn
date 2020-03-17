import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets,layers,Sequential,optimizers,metrics

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
def preprocess(x,y):
	x = tf.cast(x,dtype=tf.float32) /255.
	x = tf.reshape(x,[28*28])
	y = tf.cast(y,dtype=tf.int32)
	y = tf.one_hot(y,depth=10)
	return x,y

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()


train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.map(preprocess).shuffle(60000).batch(128)
test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data = test_data.map(preprocess).batch(128)
#
#
# network = Sequential([
# 	layers.Dense(256,'relu'),
# 	layers.Dense(128,'relu'),
# 	layers.Dense(64,'relu'),
# 	layers.Dense(32,'relu'),
# 	layers.Dense(10)
# ])
# network.build(input_shape=(None,28*28))
# network.summary()
#
# network.compile(optimizer=optimizers.Adam(lr=0.01),
# 				loss=tf.losses.CategoricalCrossentropy(from_logits=True),
# 				metrics=['accuracy']
# )
# network.fit(train_data,epochs=3,validation_data=test_data,validation_freq=2)
# print("最终测试结果")
# network.evaluate(test_data)
# #
# # network.save_weights('/home/tjk/project/python/model_save/weights.ckpt')
# # print("保存模型成功")
# # network.save('/home/tjk/project/python/model_save/weights.h5')
# # print("保存模型成功")
# tf.saved_model.save(network,'/home/tjk/project/python/model_save/model/')
# print("保存模型成功")

### 加载模型训练
# network = Sequential([layers.Dense(256, activation='relu'),
#                      layers.Dense(128, activation='relu'),
#                      layers.Dense(64, activation='relu'),
#                      layers.Dense(32, activation='relu'),
#                      layers.Dense(10)])
# network = tf.keras.models.load_model('/home/tjk/project/python/model_save/weights.h5', compile=False)
# network.compile(optimizer=optimizers.Adam(lr=0.01),
# 		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
# 		metrics=['accuracy']
# 	)q

# # network.load_weights('/home/tjk/project/python/model_save/weights.ckpt')
DEFAULT_FUNCTION_KEY = "serving_default"
loaded = tf.saved_model.load('/home/tjk/project/python/model_save/model/')
network = loaded.signatures[DEFAULT_FUNCTION_KEY]


print('加载 weights 成功')

test_image = plt.imread('/home/tjk/project/python/8.png')

plt.imshow(test_image)
x = test_image.reshape(1, 784)
print(x.shape)
plt.show()

# pred = network.predict(x)
pred = loaded(x)
print("预测结果原始结果", pred)
pred = tf.nn.softmax(pred, axis=1)
print("预测softmax后", pred)
pred = tf.argmax(pred, axis=1)
print("最终测试结果", pred)