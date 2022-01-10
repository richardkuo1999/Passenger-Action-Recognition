# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:09:18 2020

@author: Admin
"""
from PIL import Image, ImageOps
import os
from os import listdir
import random
import cv2
import numpy as np
#img.transpose(Image.FLIP_LEFT_RIGHT)

pathofpose = 'E:/3dres4class/data/winbus-06-notest'
#pathofpose = 'D:/靜態姿勢資料-新版/orisize_1203-改id跟資料夾名稱/test/cam2-10'
#pathofpose = 'E:/3dres/3D-ResNets-PyTorch-master/addlegsstand/winbus-03_twostate-notest'
#pathofpose = 'E:/3dres4class/winbus-02-3'
poses = listdir(pathofpose)

for pose in poses:
    pathofid = pathofpose + os.sep + pose
    idlist = listdir(pathofid)
    for id in idlist:
        idpath = pathofid + os.sep + id
        pa = 'E:/3dres4class/traindata/winbus-06-2RoHf'
        #pa = 'D:/靜態姿勢資料-新版/orisize_1203-改id跟資料夾名稱/test/cam2-10-2'
        kindofaug = '-4'
        os.mkdir(pa + os.sep + pose + os.sep + id + kindofaug)
        imglist = listdir(idpath)
        rotdeg = random.choice([-1,1]) * random.randint(15, 30)
        a0 = np.random.uniform(-1, 1, 3)
        a = a0 * [0.2, 0.8, 0.388 ] + 1
        #HF = random.random() < 0.5
        HF = 1
        for imgs in imglist:
            img = Image.open(idpath + os.sep + imgs)
            if HF:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
            img = img.rotate(rotdeg, fillcolor = (128, 128, 128))
            #img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            #img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * a).clip(None, 255).astype(np.uint8)
            #np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])  # inplace hue clip (0 - 179 deg)
            #cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
            #img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            img.save(pa + os.sep + pose + os.sep + id + kindofaug + os.sep + imgs)
            #img.save('E:/3dres4class/winbus-02-onlytest' + os.sep + pose + os.sep + id + os.sep + imgs)
        #print(pathofid + os.sep + id)

'''
rotdeg = random.randint(-30, 30)
img = Image.open('E:/3dres2classex/winbus-02/sitting/000000/image_00001.jpg')
img = img.rotate(rotdeg, fillcolor = (128, 128, 128))
img.save('E:/ss.jpg')
s = random.choice([-1,1])
print(random.choice([-1,1]) * random.randint(15, 30))
'''

'''
a0 = np.random.uniform(-1, 1, 3)
#a = a0 * [0.0138, 0.678, 0.36] + 1
a = a0 * [0.2, 0.8, 0.388 ] + 1
print('a0',a0)
print('before',a)
#a[0] = 1
print('after',a)
img = cv2.imread('C:/Users/Admin/Desktop/cam1-GH010034 0092.jpg')
img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * a).clip(None, 255).astype(np.uint8)
np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])  # inplace hue clip (0 - 179 deg)
cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
img = cv2.resize(img,(1280,720))
cv2.imshow('img',img)
#cv2.waitkey()
cv2.waitKey(0)
'''
