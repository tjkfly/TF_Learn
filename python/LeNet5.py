# encoding: utf-8
"""
@author: tjk
@contact: tjk@email.com
@time: 2020/3/17 下午5:41
@file: LeNet5.py
@desc: 
"""
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import matplotlib.pyplot as plt
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




def preprocess(x,y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.expand_dims(x,axis=-1)  # 维度扩展
    y = tf.cast(y, dtype=tf.int32)
    # y = tf.one_hot(y, depth=10)
    return x, y

(x, y),(x_test,y_test) = keras.datasets.mnist.load_data()
train_data = tf.data.Dataset.from_tensor_slices((x,y))
train_data = train_data.map(preprocess).batch(128).repeat(10)

test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data = test_data.map(preprocess).batch(128)


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')
        self.pool2 = keras.layers.MaxPool2D((2,2),strides=(2,2))
        self.conv3 = keras.layers.Conv2D(64,(5,5),activation='relu')
        self.pool4 = keras.layers.MaxPool2D((2,2))
        self.dropout1 = keras.layers.Dropout(0.25)
        self.flatten = keras.layers.Flatten()
        self.dense5 = keras.layers.Dense(120,activation='relu')
        self.dropout2 = keras.layers.Dropout(0.5)
        self.dense6 = keras.layers.Dense(84, activation='relu')
        self.dropout3 = keras.layers.Dropout(0.5)
        self.dense7 = keras.layers.Dense(10)
    def call(self,inputs,training=None,**kwargs):
        x = self.conv1(inputs)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool4(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense5 (x)
        x = self.dropout2(x)
        x = self.dense6(x)
        x = self.dropout3(x)
        x = self.dense7(x)
        return x

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = MyModel()
model.build(input_shape=(None,28,28,1))
model.summary()
model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
)
history = model.fit(train_data,epochs=5,validation_data=test_data,validation_freq=1,
                    callbacks=[tensorboard_callback])


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# tf.saved_model.save(model,'/home/tjk/project/model/mnist/')
# print("保存模型成功")
# # Convert Keras model to ConcreteFunction
# full_model = tf.function(lambda x: model(x))
# full_model = full_model.get_concrete_function(
#     tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
#
# # Get frozen ConcreteFunction
# frozen_func = convert_variables_to_constants_v2(full_model)
# frozen_func.graph.as_graph_def()
#
# layers = [op.name for op in frozen_func.graph.get_operations()]
# print("-" * 50)
# print("Frozen model layers: ")
# for layer in layers:
#     print(layer)
#
# print("-" * 50)
# print("Frozen model inputs: ")
# print(frozen_func.inputs)
# print("Frozen model outputs: ")
# print(frozen_func.outputs)
#
# # Save frozen graph from frozen ConcreteFunction to hard drive
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                   logdir="/home/tjk/project/model/mnist/frozen_models",
#                   name="mnist_graph.pb",
#                   as_text=False)
