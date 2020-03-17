# encoding: utf-8
"""
@author: tjk
@contact: tjk@email.com
@time: 2020/3/16 下午12:04
@file: cv_1.py
@desc: 
"""
import cv2
from cv2 import dnn
import numpy as np

print(cv2.__version__)


class_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
img = cv2.imread('/home/tjk/project/picture/aj1.png', cv2.IMREAD_GRAYSCALE)
# img_cv2 = cv2.cvtColor(img_cv2,cv2.COLOR_BGR2GRAY)

cv2.imshow("2",img)
print(img.shape)
# print(1-img_cv21/255.)
blob = cv2.dnn.blobFromImage(1-img,
                             scalefactor=1.0/225.,
                             size=(28, 28),
                             mean=(0, 0, 0),
                             swapRB=False,
                             crop=False)

print("[INFO]img shape: ", blob.shape)

net = dnn.readNetFromTensorflow('/home/tjk/project/tf_doc/frozen_models/frozen_graph.pb')

print("success!!")
net.setInput(blob)
out = net.forward()
out = out.flatten()

classId = np.argmax(out)
print("classId",classId)
print("预测结果为：",class_name[classId])
cv2.waitKey(0)




# import tensorflow as tf
# from tensorflow import keras
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# font = cv2.FONT_HERSHEY_SIMPLEX
# model =keras.models.load_model('my_model.h5') #读取网络
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# image_path = "z.png"#在这里写入图片路径
# def look_image(data):
#     plt.figure()
#     data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
#     plt.imshow(data)
# image = cv2.imread(image_path)#读取图片
# image_ =  cv2.resize(image, (250,250), interpolation = cv2.INTER_AREA)
# image = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)#灰度化处理
# img_w = cv2.Sobel(image,cv2.CV_16S,0,1)#Sobel滤波，边缘检测
# img_h = cv2.Sobel(image,cv2.CV_16S,1,0)#Sobel滤波，边缘检测
# img_w = cv2.convertScaleAbs(img_w)
# _, img_w = cv2.threshold(img_w,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
# img_h = cv2.convertScaleAbs(img_h)
# _, img_h = cv2.threshold(img_h,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
# image = img_w + img_h
# image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
# temp_data = np.zeros((250,10))
# image = np.concatenate((temp_data,image,temp_data),axis = 1)
# temp_data = np.zeros((10,270))
# image = np.concatenate((temp_data,image,temp_data),axis = 0)
# image = cv2.convertScaleAbs(image)
# contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for _ in contours:
#     x,y,w,h = cv2.boundingRect(_)
#     if w*h  < 100:
#         continue
#     img_model = image[y-10:y+h+10,x-10:x+w+10]
#     img_model =  cv2.resize(img_model, (28,28), interpolation = cv2.INTER_AREA)
#     img_model = img_model/255
#     predict = model.predict(img_model.reshape(-1,28,28,1))
#     if np.max(predict) > 0.5:
#         data_predict = str(np.argmax(predict))
#         image_z = cv2.rectangle(image_,(x-10,y-10),(x + w-10,y + h-10),(255,0,0),1)
#         image_z = cv2.putText(image_z,data_predict , (x+10, y+10), font, 0.7, (0, 0, 255), 1)
# look_image(image_z)
# save = cv2.imwrite( "image_predict2.png",image_z)
