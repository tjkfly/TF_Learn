# encoding: utf-8
"""
@author: tjk
@contact: tjk@email.com
@time: 2020/3/14 下午6:38
@file: cifar10_1.py
@desc: 
"""
import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras import Sequential,datasets,layers,models
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

(image_train,label_train),(image_test,label_test) = datasets.cifar10.load_data()
image_train ,image_test = image_train/255. , image_test /255.
print("image_train:",image_train.shape,"label_train:",label_train.shape)

class_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i],cmap=plt.cm.binary)
    plt.xlabel(class_name[label_train[i][0]])
plt.show()


model = Sequential([
    layers.Conv2D(32,(3,3),activation='relu',name='conv1'),
    layers.MaxPool2D((2,2),name='pool1'),
    layers.Conv2D(64,(3,3),activation='relu',name='conv2'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10)
])
model.build(input_shape=(None,32,32,3))
model.summary()

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history = model.fit(image_train,label_train,epochs=1,validation_data=(image_test,label_test))
