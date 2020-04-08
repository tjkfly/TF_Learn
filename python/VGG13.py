# encoding: utf-8
"""
@author: tjk
@contact: tjk@email.com
@time: 2020/3/19 下午7:52
@file: VGG13.py
@desc:
"""
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)
net_layers = [
    #unit1
    keras.layers.Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'),
    keras.layers.Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
    # unit2
    keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),

    # unit3
    keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),

    # unit4
    keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    # unit5
    keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    #flatten
    keras.layers.Flatten(),
    #fc_net
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(100)
]

def preprocess(x, y):
    x = tf.cast(x,dtype=tf.float32) / 255.
    y = tf.cast(y,dtype=tf.int32)
    return x, y

(x, y),(x_test ,y_test) = keras.datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape,x_test.shape ,y_test.shape)



#batch 大小影响 loss 下降的速度
train_data = tf.data.Dataset.from_tensor_slices((x,y))
train_data = train_data.map(preprocess).batch(128)

test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data = test_data.map(preprocess).batch(128)

def main():
    model = keras.Sequential(net_layers)
    model.build(input_shape=[None,32,32,3])
    model.summary()
    # x = tf.random.normal([4,32,32,3])
    # y = model(x)
    # print(y.shape)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_data,epochs=15,validation_data=test_data,validation_freq=2)
    model.evaluate(test_data)
if __name__ == '__main__':
    main()
