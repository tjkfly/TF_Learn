# encoding: utf-8
"""
@author: tjk
@contact: tjk@email.com
@time: 2020/3/29 下午3:47
@file: nms.py
@desc: 
"""
import cv2
import numpy as np

#
dets = np.array([[204,102,358,250,0.5],
                 [257,118,380,250,0.7],
                 [280,135,400,250,0.6],
                 [255,118,360,235,0.7]])
def nms(dets,thresh):
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    scores = dets[:,4]
    areas = (x2 - x1 +1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    print(type(order))
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] -inter)
        print(overlap)

        inds = np.where(overlap <= thresh)[0]
        print("inds",inds)

        order = order[inds + 1]
        print("order",order)
    return keep
print(nms(dets,0.5))