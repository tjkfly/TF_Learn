# encoding: utf-8
"""
@author: tjk
@contact: tjk@email.com
@time: 2020/3/26 下午9:47
@file: test1.py
@desc: 
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
print(tf.__version__)

tfds.disable_progress_bar()
(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
print(raw_train)
print(raw_validation)
print(raw_test)

# 显示训练集中的前两个图像和标签：
get_label_name = metadata.features['label'].int2str
for  image , label in raw_train.take(2):
    plt.figure()
    plt.imshow()
    plt.title(get_label_name(label))
# 格式化数据
"""
使用该tf.image模块格式化任务的图像。
将图像调整为固定的输入尺寸，然后将输入通道调整为一定范围 [-1,1]
"""
IMG_SIZE = 160
def format_example(image,label):
    image = tf.cast(image,tf.float32)
    image = (image/127.5) -1
    image = tf.image.resize(image,(IMG_SIZE,IMG_SIZE))
    return image, label
# 使用map方法将此函数应用于数据集中的每个项目：

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

#检查一批数据：
image_batch =train_batches.take(1)

"""
首先，实例化一个预加载了ImageNet训练权重的MobileNet V2模型。
通过指定include_top = False参数，可以加载不在顶部包括分类层的网络，这对于特征提取是理想的。
"""
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
#该特征提取器将每个160x160x3图像转换为一个5x5x1280特征块。看看它对示例图像批次有什么作用：
feature_batch = base_model(image_batch)
print(feature_batch.shape)

#特征提取
#在此步骤中，将冻结从上一步创建的卷积基础并将其用作特征提取器。此外，您可以在其顶部添加分类器并训练顶级分类器。

base_model.trainable = False
base_model.summary()