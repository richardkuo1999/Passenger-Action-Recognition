#-*- coding: utf-8 -*-
import os
import numpy as np

from computeMap.voc_eval import voc_eval   # 注意將voc_eval.py和compute_mAP.py放在同一級目錄下

detpath = 'detres/'   # 各類txt文件路徑
#detpath = 'eval-2-b7/'   # 各類txt文件路徑
detfiles = os.listdir(detpath)

classes = ('__background__', # always index 0 數據集類別
                  'sitting', 'standing', 'sit', 'stand')
#classes = ('__background__', 'sitting', 'sit')

aps = []      # 保存各類ap
recs = []     # 保存recall
precs = []    # 保存精度

annopath = 'computeMap/cam1-10_frameap/' + '{:s}.xml'    # annotations的路徑，{:s}.xml方便後面根據圖像名字讀取對應的xml文件
imagesetfile = 'computeMap/filenames.txt'  # 讀取圖像名字列表文件
cachedir = 'computeMap/cache/'

for i, cls in enumerate(classes):
    if cls == '__background__':
        continue
    for f in detfiles:    # 讀取cls類對應的txt文件
        #print(f, cls)
        #print(f.find(cls))
        if f.split('.')[0] == cls:
            filename = detpath + f
            #print('detfiles:',detfiles)
            #print('filename:',filename)
            rec, prec, ap, tp, fp, npos, tol_diff, tol_nodiff = voc_eval(        # 調用voc_eval.py計算cls類的recall precision ap
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0,
                use_07_metric=False)

    aps += [ap]
    print('---------------------')
    print('AP for {} = {:.4f}'.format(cls, ap))
    print('recall for {} = {:.4f}'.format(cls, rec[-1]))
    print('precision for {} = {:.4f}'.format(cls, prec[-1]))
    print('tp:',tp, 'fp:',fp, 'nodiff:',npos, 'total diff:',tol_diff, 'nodiff:',tol_nodiff)
    print('cls_res: %f'%(tp / float(npos)))

print('Mean AP = {:.4f}'.format(np.mean(aps)))
print('~~~~~~~~')

print('Results:')
for ap in aps:
    print('{:.3f}'.format(ap))
print('{:.3f}'.format(np.mean(aps)))
print('~~~~~~~~')