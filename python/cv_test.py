# -*- coding: utf-8 -*-
# @Author: tjk
# @Date:   2020-03-10 22:26:20
# @Last Modified by:   tjk
# @Last Modified time: 2020-03-11 10:36:25
import matplotlib.pyplot as plt 
import cv2
image = cv2.imread("/home/tjk/project/python/7.png")
# ima3 = plt.imread("/home/tjk/project/python/7.png")
cv2.imshow("dfdf",image)
 
# plt.imshow(ima3)
# xx = ima3.reshape(1,784)
# print(xx.shape)
cv2.waitKey(0)
# plt.show()