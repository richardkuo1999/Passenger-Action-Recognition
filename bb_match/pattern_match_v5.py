# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 01:13:38 2020

@author: Admin

前版v4使用模型預測X,Y,W,H位置範圍是0~99
v5直接預測位置範圍0~1,損失函數為GIOU LOSS
"""
import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from network import *
from datasets_v2 import *
from utils.utils import *
from opts import parse_opts
from ioumatch import matching2

def train(epoch, criterion):
    print('train in epoch %d'%epoch)
    net.train()
    train_loss = 0.
    meaniou = 0.
    numiou1 = 0.
    numiou2 = 0.
    numiou3 = 0.
    numdata = 0.
    nb = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
    for i, (inputs, targets, id) in pbar:
        inputs = inputs.to(device).float()
        inputs = Variable(inputs, requires_grad=True)
        targets = targets.to(device).float()
        #targets = Variable(targets, requires_grad=True)
        tarlen = len(targets)
        targets = torch.reshape(targets, (int(tarlen/4), 4))
        optimizer.zero_grad()
        output = net(inputs) # y,x,w,h
        loss = criterion(output, targets) # mse loss
        giou, iou = bbox_iou(output.t(), targets)
        #loss = (1.0 - giou).mean() # giou loss
        meaniou += iou.mean()
        numiou1 += (iou > 0.1).sum().item()
        numiou2 += (iou > 0.3).sum().item()
        numiou3 += (iou > 0.5).sum().item()
        numdata += len(targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('\ntrain loss: %.4f'%train_loss)
    print('iou mean: ', meaniou/nb)
    print('ratio iou: ', numiou1/numdata, numiou2/numdata, numiou3/numdata)
    return train_loss, meaniou.item()/nb, numiou1/numdata, numiou2/numdata, numiou3/numdata

def bbox_iou(box1, box2): # ([x1,y1,w1,h1].t(), [x2,y2,w2,h2])
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c_area = cw * ch + 1e-16  # convex area
    return iou - (c_area - union) / c_area, iou  # GIoU, IoU

def calRLbbiou(nptar, nppred, img_w, img_h):
    nptar_y, nptar_w, nptar_h = nptar
    nppred_y, nppred_w, nppred_h = nppred
    gtHeight = round(nptar_w*0.01*img_w)
    gtWidth = round(nptar_h*0.01*img_h)
    bbHeight = round(nppred_w*0.01*img_w)
    bbWidth = round(nppred_h*0.01*img_h)
    gty1 = round(nptar_y*0.01*img_w-(gtWidth/2))
    gty2 = gty1+gtWidth
    bby1 = round(nppred_y*0.01*img_w-(bbWidth/2))
    bby2 = bby1+bbWidth

    endx = max(gtWidth,bbWidth)
    startx = 0
    width_inter = gtWidth+bbWidth-(endx-startx)

    endy = max(gty1+gtHeight,bby1+bbHeight)
    starty = min(gty1,bby1)
    height_inter = gtHeight+bbHeight-(endy-starty)
    if width_inter <=0 or height_inter <=0 :
        ratio = 0
    else:
        Area_inter = width_inter*height_inter
        Area1 = gtHeight*gtWidth
        Area2 = bbHeight*bbWidth
        ratio = Area_inter*1./(Area1+Area2-Area_inter)
    return ratio

def val(epoch, criterion):
    print('val in epoch %d'%epoch)
    net.eval()
    val_loss = 0.
    meaniou = 0.
    numiou1 = 0.
    numiou2 = 0.
    numiou3 = 0.
    numdata = 0.
    nb = len(testdataloader)
    pbar = tqdm(enumerate(testdataloader), total=nb)  # progress bar
    for i, (inputs, targets, id) in pbar:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        tarlen = len(targets)
        targets = torch.reshape(targets, (int(tarlen/4), 4))
        output = net(inputs) # y,x,w,h
        val_loss = criterion(output, targets) #mse loss
        giou, iou = bbox_iou(output.t(), targets)
        #val_loss = (1.0 - giou).mean() # giou loss
        meaniou += iou.mean()
        numiou1 += (iou > 0.1).sum().item()
        numiou2 += (iou > 0.3).sum().item()
        numiou3 += (iou > 0.5).sum().item()
        numdata += len(targets)
    print('\nval loss: %.4f'%val_loss)
    print('iou mean: ', meaniou/nb)
    print('ratio iou: ', numiou1/numdata, numiou2/numdata, numiou3/numdata)
    return val_loss, meaniou.item()/nb, numiou1/numdata, numiou2/numdata, numiou3/numdata

def test_1frame(inputs, targets, softmax, img_w, img_h):
    inputs = inputs.to(device).float()
    tarlen = len(targets)
    targets_content = targets.clone()
    targets = targets * 100
    targets = targets.to(device, dtype=torch.int64)
    targets_y = targets[0:tarlen:4]
    targets_w = targets[2:tarlen:4]
    targets_h = targets[3:tarlen:4]
    nptar_y = targets_y.cpu().numpy()
    nptar_w = targets_w.cpu().numpy()
    nptar_h = targets_h.cpu().numpy()  

    print('targets:')
    print(targets_y,targets_w,targets_h)
    
    outputs_y, outputs_w, outputs_h = net(inputs)     
    nppred_y = predicted_y.cpu().numpy()
    nppred_w = predicted_w.cpu().numpy()
    nppred_h = predicted_h.cpu().numpy()

    print('pred:')
    print(predicted_y,predicted_w,predicted_h)

    profit_matrix = [] 
    for i, pred in enumerate(predicted_y):
        pro = []
        for x in range(len(targets_y)):
            nptar = (nptar_y[x], nptar_w[x], nptar_h[x])
            nppred = (nppred_y[i], nppred_w[i], nppred_h[i])
            iouValue = calRLbbiou(nptar, nppred, img_w, img_h)
            pro.append(iouValue) # Cost function 4
        profit_matrix.append(pro)
    #profit_matrix = torch.tensor(profit_matrix)
    #profit_matrix = profit_matrix.detach().cpu().numpy()
    print('profit_matrix:')
    print(profit_matrix)
    match_res = matching2(profit_matrix) # Do Maximum matching
    print('match_res:')
    print(match_res)
    correct = 0.
    total = 0.
    for i in range(len(match_res)):
        matched1 = inputs[match_res[i][0]] # Input bb in match results
        matched_id = targets_y[match_res[i][1]] # Corresponding bb of target id in match results
        matched_content = targets_content[match_res[i][1]*4:match_res[i][1]*4+4] # Corresponding bb of target content in match results
        if matched_id == targets_y[match_res[i][0]]:
            correct += 1
        total += 1
        print(matched1.cpu().numpy(), ' and ', matched_content.cpu().numpy())
    return correct/total

def test_epoch(epoch):
    print('test in epoch %d'%epoch)
    net.eval()
    softmax = nn.Softmax()
    nb = len(testdataloader)
    pbar = tqdm(enumerate(testdataloader), total=nb)  # progress bar
    pre_id = 0
    datain = None
    datatar = None
    dataall = 0.
    acc = 0.
    img_w = 2704
    img_h = 1520
    err_list = []
    for i, (inputs, targets, frameid) in pbar:
        if frameid == pre_id:
            if datain is not None:              
                datain = torch.cat((datain,inputs),0)
                datatar = torch.cat((datatar,targets),0)
            else:
                datain = inputs
                datatar = targets
        else:
            if len(datain) >= 2:
                print('---------------')
                print('test frame id:', frameid-1)
                accin1frame = test_1frame(datain, datatar, softmax, img_w, img_h)
                if accin1frame != 1:
                    err_list.append(frameid-1)
                acc += accin1frame
                datain = inputs
                datatar = targets
                dataall += 1
            else:
                print('ignored frame id:', frameid-1)
                datain = inputs
                datatar = targets
        pre_id = frameid

    if len(datain) >= 2:
        print('---------------')
        print('test frame id:', frameid)
        accin1frame = test_1frame(datain, datatar, softmax, img_w, img_h)
        if accin1frame != 1:
            err_list.append(frameid)
        acc += accin1frame
        dataall += 1
    print('accuracy=', acc/dataall)
    print('all data number:', dataall)
    print('error frame:')
    print(err_list)
    return


if __name__ == '__main__':
    opt = parse_opts()
    print(opt)
    epochs = opt.n_epochs
    lr = opt.lr
    batch_size = opt.batch_size
    lr_patience = opt.lr_patience
    train_path = opt.train_path
    test_path = opt.test_path
    no_train = opt.no_train
    no_val = opt.no_val
    test = opt.test
    resume_path = opt.resume_path
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('weight'):
        os.mkdir('weight')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    init_seeds()
    net = Net6().to(device)
    
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss(reduction='mean')

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    #scheduler = lr_scheduler.ReduceLROnPlateau(
    #            optimizer, 'min', patience=lr_patience)
    tmax = opt.n_epochs + 1
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = tmax)

    dataset = LoadImagesAndLabels(train_path, batch_size)
    dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=0,
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)
    testdataset = LoadImagesAndLabels(test_path, 1)
    testdataloader = DataLoader(testdataset,
                                batch_size=batch_size,
                                #num_workers=opt.num_workers,
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)
    begin_epoch = 0
    val_epoch = 0
    if resume_path:
        print('loading checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        begin_epoch = checkpoint['epoch']
        print("begin_epoch",begin_epoch)
        val_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        if not no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    f = open('results/results.txt','a')
    line = 'epoch    val loss    mean iou    iou > 0.1    iou > 0.3    iou > 0.5    train iou > 0.5    lr\n'
    f.write(line)
    f.close()

    f = open('results/results_train.txt','a')
    line = 'epoch    train loss    mean iou    iou > 0.1    iou > 0.3    iou > 0.5    lr\n'
    f.write(line)
    f.close()

    train_best_iou = 0.
    val_best_iou = 0.
    t_numiou3 = 0.
    v_numiou1 = 0.
    v_numiou2 = 0.
    v_numiou3 = 0.
    v_loss = 0.
    v_meaniou = 0.
    for epoch in range(begin_epoch, epochs):

        if not no_train:
            t_loss, t_meaniou, t_numiou1, t_numiou2, t_numiou3 = train(epoch, criterion)
            scheduler.step()
            if t_numiou3 >= train_best_iou: train_best_iou = t_numiou3

        if not no_val:
            v_loss, v_meaniou, v_numiou1, v_numiou2, v_numiou3 = val(epoch, criterion)
            if v_numiou3 >= val_best_iou: val_best_iou = v_numiou3

        print('train best: ', train_best_iou, ' val best: ', val_best_iou)
        f = open('results/results.txt','a')
        line = ('%d\t' + '%.4f       '*6 + '%5g\n')%(epoch,
            v_loss,
            v_meaniou,
            v_numiou1,
            v_numiou2,
            v_numiou3,
            t_numiou3,
            optimizer.param_groups[0]['lr'])
        f.write(line)
        f.close()

        f = open('results/results_train.txt','a')
        line = ('%d\t' + '%.4f       '*5 + '%5g\n')%(epoch,
            t_loss,
            t_meaniou,
            t_numiou1,
            t_numiou2,
            t_numiou3,
            optimizer.param_groups[0]['lr'])
        f.write(line)
        f.close()

        if v_numiou3 >= val_best_iou:
                save_file_path = './weight/best.pth'
                states = {
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)

    if test:
        print('testing')
        test_epoch(begin_epoch)