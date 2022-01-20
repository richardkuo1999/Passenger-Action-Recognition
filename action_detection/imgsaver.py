# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:56:50 2020

@author: Admin
"""

import torch
import numpy as np
from torchvision.utils import save_image
import cv2
import os

# 儲存所有人的IMG
class Saved_imgs(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.det_imgs = dict()
    
    # 每個FRAME CAT IMG後刪除沒追蹤到的ID IMG
    def checkdel(self, cur_id):
        all_id = list(self.det_imgs.keys())
        for id in all_id:
            if not id in cur_id:
                #print('delete id:', id)
                del self.det_imgs[id]
    
    # 一個ID增加IMG
    def update(self, new_img, id, indexx):
        #如果CAT的IMG大於一定數量 刪除最舊的IMG(INDEX=0)
        if id in self.det_imgs:
            if len(self.det_imgs[id]) == 15:
                '''
                dir = 'ssss/%06d'%(indexx)
                #dir = 'ssss/%06d'%(id)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                for i in range(0, 15):
                    #imgname = 'ssss/%06d'%(id) + '/image_%05d'%(i+1) + '.jpg'
                    imgname = 'ssss/%06d'%(indexx) + '/image_%05d'%(i+1) + '.jpg'
                    img = self.det_imgs[id][i].clone()
                    save_image(img, imgname)
                '''
                self.det_imgs[id] = torch.cat((self.det_imgs[id][1:], new_img.unsqueeze(0)),0) # Work code
                #self.det_imgs[id] = self.det_imgs[id][14:] # For crop image

            else: self.det_imgs[id] = torch.cat((self.det_imgs[id], new_img.unsqueeze(0)),0)
            #self.det_imgs[id] = self.det_imgs[id].permute(1, 0, 2, 3)
        else:
            self.det_imgs[id] = new_img.unsqueeze(0)
            #print('add new key:', id)

    def checklen(self, id):
        if self.det_imgs[id].size()[0] == 15:
            #print('inlen')
            #print(np.shape(self.det_imgs[id].permute(1, 0, 2, 3).unsqueeze(0)))
            #return self.det_imgs[id].permute(1, 0, 2, 3).unsqueeze(0)
            return self.det_imgs[id].permute(1, 0, 2, 3).unsqueeze(0)

if __name__ == '__main__':
    img = cv2.imread('wallpaper.png')
    img = cv2.resize(img,(4,4),1)
    img = torch.tensor(img)

    img2 = cv2.imread('cam1-GH010034 0092.jpg')
    img2 = cv2.resize(img2,(4,4),1)
    img2 = torch.tensor(img2)

    temp_imgs = Saved_imgs()
    t1_id = 1
    temp_imgs.update(img,t1_id)
    temp_imgs.update(img,t1_id)
    temp_imgs.update(img,t1_id)
    temp_imgs.update(img,t1_id)
    temp_imgs.update(img,t1_id)
    temp_imgs.update(img,t1_id)

    #t2_id = 2
    #temp_imgs.update(img2,t2_id)
    #temp_imgs.update(img2,t2_id)
    '''
    t3_id = 3
    temp_imgs.update(img2,t3_id)
    temp_imgs.update(img2,t3_id)
    '''
            
    '''
    # 測試刪除最舊IMG
    print('test del oldest img')
    print(a1)
    temp_imgs.update(img2,t1_id)
    print(a1)
    print('--------------------')
    '''


    # 如果IMG長度一致 一個BATCH丟到分類器
    a = temp_imgs.det_imgs
    a1 = a[1].unsqueeze(0)
    #a2 = a[2].unsqueeze(0)
    #a3 = a[3].unsqueeze(0)
    #aa = torch.cat((a1,a2),0)
    #aa = torch.cat((aa,a3),0)
    #print(a1)
    print(np.shape(a1))
