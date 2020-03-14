# -*- coding: utf-8 -*-
# @Author: tjk
# @Date:   2020-03-10 19:50:57
# @Last Modified by:   tjk
# @Last Modified time: 2020-03-10 23:23:48
import tensorflow as tf 
import os
from tensorflow import keras
from tensorflow.keras import datasets,layers,Sequential,optimizers,metrics
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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


network = Sequential([
	layers.Dense(256,'relu'),
	layers.Dense(128,'relu'),
	layers.Dense(64,'relu'),
	layers.Dense(32,'relu'),
	layers.Dense(10)
])
network.build(input_shape=(None,28*28))
network.summary()

network.compile(optimizer=optimizers.Adam(lr=0.01),
				loss=tf.losses.CategoricalCrossentropy(from_logits=True),
				metrics=['accuracy']
)
network.fit(train_data,epochs=5,validation_data=test_data,validation_freq=2)
print("最终测试结果")
network.evaluate(test_data)







########预测
x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)/255.
x = tf.reshape(x_test[0],[-1,28*28])
print("测试图片的标签为：",y_test[0])


ima3 = plt.imread("/home/tjk/project/python/7.png")

 
plt.imshow(ima3)
xx = ima3.reshape(1,784)
print(xx.shape)

plt.show()


pred = network.predict(xx)
print("预测结果原始结果",pred)
pred = tf.nn.softmax(pred, axis=1)
print("预测softmax后",pred)
pred = tf.argmax(pred, axis=1)
print("最终测试结果",pred)



# image = x_test[0].numpy()
# image =image.reshape(28,28)
# print(image.shape)
# plt.imshow(image)
# plt.show()

