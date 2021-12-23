# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:04:56 2019

@author: flove
"""

import numpy as np
from hungarian import *
import time
import random

VIEW_MATCH_RESULT = False
VIEW_MATCH_TIME = False

#前frame
xyxy1 = np.array([
        #id
        [1, 104, 84, 359, 336], #BB1 sit
        [50, 706, 464, 953, 712], #BB3 stand
        [3, 852, 342, 1111, 588], #BB2 stand
        [2, 231, 25, 475, 246], #BB0 sit
        ])
#當前frame
xyxy2 = np.array([
        #id
        [0, 220, 7, 521, 279], #GT0 sit
        [1, 81, 110, 385, 388], #GT1 sit
        [2, 768, 280, 1101, 600], #GT2 stand
        [3, 638, 480, 978, 781] #GT3 stand
        ]) 

def cal_iou(xyxy1,xyxy2): #xyxy: cls, x1, y1, x2, y2
    match_list = [0] *len(xyxy2)
    for i in range(0,len(xyxy2)):
        # Compute overlaps
        # Intersection
        ixmin = np.maximum(xyxy1[:, 1], xyxy2[i,1])
        iymin = np.maximum(xyxy1[:, 2], xyxy2[i,2])
        ixmax = np.minimum(xyxy1[:, 3], xyxy2[i,3])
        iymax = np.minimum(xyxy1[:, 4], xyxy2[i,4])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        # union
        uni = ((xyxy2[i,3] - xyxy2[i,1] + 1.) * (xyxy2[i,4] - xyxy2[i,2] + 1.) +
               (xyxy1[:, 3] - xyxy1[:, 1] + 1.) *
               (xyxy1[:, 4] - xyxy1[:, 2] + 1.) - inters) 
        overlaps = inters / uni
        #ovmax = np.max(overlaps)
        #jmax = np.argmax(overlaps)
        
        #以IOU高低做匈牙利匹配
        match_list[i] = overlaps
        
    return match_list

def matching(xyxy1, xyxy2):
    #print('\nxyxy1:',xyxy1)
    #print('\nxyxy2:', xyxy2)
    start_t = time.time()
    match_list = cal_iou(xyxy1, xyxy2)
    print(match_list)
    hungarian = Hungarian(match_list,is_profit_matrix=True)
    hungarian.calculate()
    match_res = hungarian.get_results() #(xyxy2,xyxy1)
    # match_res[i][0] is 當前frame id, match_res[i][1] is 前frame id
    res = []
    '''
    for i in range(0,len(xyxy2)):
        if VIEW_MATCH_RESULT:
            print('result: ', xyxy2[match_res[i][0]], 'match: ', xyxy1[match_res[i][1]])
        if VIEW_MATCH_TIME:
            print('spent time: ', time.time()-start_t)
    '''
    check = [0]*len(xyxy2)
    for x in range(0,len(xyxy2)):
        for xx in range(0,len(match_res)):
            if x==match_res[xx][0]:
                xyxy2[x][0]=xyxy1[match_res[xx][1]][0]
                check[x]=1
    for x in range(0,len(check)):
        if check[x]!=1:
            xyxy2[x][0] = random.randint(50,999)

    return xyxy2

def matching2(xyxy):
    #print(np.shape(xyxy))
    hungarian = Hungarian(xyxy,is_profit_matrix=True)
    hungarian.calculate()
    match_res = hungarian.get_results()
    res = []
    return match_res

if __name__ == '__main__':
    gg = matching(xyxy1, xyxy2)
    print(gg)