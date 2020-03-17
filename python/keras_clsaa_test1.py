# -*- coding: utf-8 -*-
# @Author: tjk
# @Date:   2020-03-11 19:53:01
# @Last Modified by:   tjk
# @Last Modified time: 2020-03-11 21:18:18
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
	"""docstring for MyDense"""
	def __init__(self, input_dim,output_dim):
		super(MyDense, self).__init__()
		self.kernel = self.add_variable('w',[input_dim,output_dim])
		self.bias = self.add_variable('b',[output_dim])
	def call(self,inputs,training=None):
		out = inputs @ self.kernel + self.bias
		return out

class MyModel(keras.Model):
	def __init__(self):
		super(MyModel,self).__init__()
		self.fc1 = MyDense(28*28,256)
		self.fc2 = MyDense(256,128)
		self.fc3 = MyDense(128,64)
		self.fc4 = MyDense(64,32)
		self.fc5 = MyDense(32,10)
	def call(self,inputs,training=None):
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
network.compile(optimizer=optimizers.Adam(lr=0.01),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
)

network.fit(train_data, epochs=5, validation_data=test_data,validation_freq=2)






