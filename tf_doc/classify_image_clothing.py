# encoding: utf-8
"""
@author: tjk
@contact: tjk@email.com
@time: 2020/3/15 下午3:50
@file: classify_image_clothing.py
@desc: 官网代码
"""
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
print('tf版本：',tf.__version__)
# 加载数据集
def prepprocess(x,y):
    # 归一化
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y
def train():
    # 构建网络
    network = keras.Sequential([
        keras.layers.Dense(128,'relu'),
        keras.layers.Dense(10)
    ])
    network.build(input_shape=(None,28*28))
    network.summary()

    network.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                  loss = tf.losses.CategoricalCrossentropy(from_logits=True),
                  # loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),      # 用这个不用tf.one_hot()
                  metrics=['accuracy']
    )
    # 训练
    history = network.fit(train_data,epochs=15,validation_data=test_data,validation_freq=1)
    plt.plot(history.history['accuracy'],label='accuracy')
    plt.plot(history.history['val_accuracy'],label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim([0.5,1])
    plt.legend(loc='lower right')
    plt.show()
    tf.saved_model.save(network,'/home/tjk/project/tf_doc/model_save/fashion_10/')
    print("保存模型成功")





    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: network(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(network.inputs[0].shape, network.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="frozen_graph.pb",
                      as_text=False)


def model_predict():
    # 预测
    DEFAULT_FUNCTION_KEY = "serving_default"
    loaded = tf.saved_model.load('/home/tjk/project/tf_doc/model_save/fashion_10/')
    network = loaded.signatures[DEFAULT_FUNCTION_KEY]
    print('加载成功')
    xx = test_image[2].reshape(1, 784)
    xx = tf.convert_to_tensor(xx, dtype=tf.float32) / 255.
    print("测试数据维度", xx.shape)

    pic = plt.imread('/home/tjk/project/picture/aj2.png')
    plt.imshow(1- pic)
    x = 1- pic
    cv2.imwrite('/home/tjk/project/picture/aj3.png', x)
    x = 1 - pic.reshape(1, 784)
    print(x.shape)
    plt.show()

    # pred = network(x)
    pred = loaded(x)
    print(type(pred))
    pred = tf.nn.softmax(pred)
    print("预测输出：", class_name[int(tf.argmax(pred, axis=1))])

    print("实际标签：", class_name[test_label[1]])

if __name__ == '__main__':
    (train_image, train_label), (test_image, test_label) = keras.datasets.fashion_mnist.load_data()
    print("数据维度(训练集)：", train_image.shape, train_label.shape)
    print("数据维度(测试集)：", test_image.shape, test_label.shape)

    train_data = tf.data.Dataset.from_tensor_slices((train_image, train_label))
    train_data = train_data.map(prepprocess).batch(128)

    test_data = tf.data.Dataset.from_tensor_slices((test_image, test_label))
    test_data = test_data.map(prepprocess).batch(128)
    class_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    # 数据可视化
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_image[i], cmap=plt.cm.binary)
        plt.xlabel(class_name[train_label[i]])
    plt.show()

    plt.figure(figsize=(5, 5))

    # train()
    # model_predict()

