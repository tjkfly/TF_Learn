import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, Sequential, optimizers, metrics

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# def preprocess(x, y):
#     x = tf.cast(x, dtype=tf.float32) / 255.
#     x = tf.reshape(x, [28 * 28])
#     y = tf.cast(y, dtype=tf.int32)
#     y = tf.one_hot(y, depth=10)
#     return x, y
#
#
# (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
#
# train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_data = train_data.map(preprocess).shuffle(60000).batch(128)
# test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# test_data = test_data.map(preprocess).batch(128)
#
# network = Sequential([
#     layers.Dense(256, 'relu'),
#     layers.Dense(128, 'relu'),
#     layers.Dense(64, 'relu'),
#     layers.Dense(32, 'relu'),
#     layers.Dense(10)
# ])
# network.build(input_shape=(None, 28 * 28))
# network.summary()
#
# network.compile(optimizer=optimizers.Adam(lr=0.01),
#                 loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy']
#                 )
# network.fit(train_data, epochs=15, validation_data=test_data, validation_freq=2)
# print("最终测试结果")
# # network.evaluate(test_data)
# # network.save_weights('/home/tjk/project/python/model_save/weights.ckpt')
# # print("保存模型成功")
# # network.save('/home/tjk/project/python/model_save/weights.h5')
# # print("保存模型成功")





def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.map(preprocess).shuffle(60000).batch(128)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.map(preprocess).batch(128)

class MyDense(layers.Layer):

    def __init__(self, num_outputs, **kwargs):
        super(MyDense, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        # self.kernel = self.add_variable("kernel",shape=[int(input_shape[-1]),self.num_outputs])
        self.kernel = self.add_weight("kernel",shape=[int(input_shape[-1]),self.num_outputs])
        self.bias = self.add_weight("kernel",shape=[self.num_outputs])
    def call(self, inputs, **kwargs):
        return  inputs @ self.kernel + self.bias
        #tf.matmul(inputs, self.kernel)



class MyModel(keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.fc1 = MyDense(256)
        self.fc2 = MyDense(128)
        self.fc3 = MyDense(64)
        self.fc4 = MyDense(32)
        self.fc5 = MyDense(10)
    def call(self,inputs,training=None,**kwargs):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x

network = MyModel()
network.build(input_shape=(None,28*28))
network.summary()
network.compile(optimizer=optimizers.Adam(lr=0.01),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
)

network.fit(train_data, epochs=5, validation_data=test_data,validation_freq=2)




tf.saved_model.save(network,'/home/tjk/project/python/model_save/model/')
print("保存模型成功")
