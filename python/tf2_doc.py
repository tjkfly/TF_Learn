# import tensorflow as tf
# from tensorflow import keras
#
# print(tf.version.VERSION)
#
#
#
#
#
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#
# train_labels = train_labels[:1000]
# test_labels = test_labels[:1000]
#
# train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
# test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.
# print("加载数据")
# # 定义一个简单的序列模型
# def create_model():
#     model = tf.keras.models.Sequential([
#         keras.layers.Dense(512, activation='relu', input_shape=(784,)),
#         keras.layers.Dropout(0.2),
#         keras.layers.Dense(10, activation='softmax')])
#
#     model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#     return model
#
# # 创建一个基本的模型实例
# model = create_model()
#
# # 显示模型的结构
# model.summary()
#
# model.fit(train_images, train_labels, epochs=5)
#
#
#
#
#
# tf.saved_model.save(model,'/home/tjk/project/python/model_save/model/')
# print("保存模型成功")

