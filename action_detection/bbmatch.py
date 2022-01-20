import torch
import torch.nn as nn
from ioumatch import matching2

def bbox_iou(box1, box2):
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
    return iou.cpu().numpy()

def test_1frame(net, inputs, targets, softmax, img_w, img_h, showid):
    '''
    inputs: xywh in 0~1
    targets: xywh in 0~1
    '''
    if showid:
        inputid = []
        tarid = []
        for i in range(0, len(inputs)): inputid.append(i)
        for i in range(0, len(targets)): tarid.append(i)

    outputs = net(inputs)     

    # Calculate cost matrix of y position
    '''
    Example: predicted->(4,3,2)
    targets(true right side of bounding box)->(5,3,2)
    Calculate 4->5, 4->3, 4->2, 3->5...
    '''
    profit_matrix = [] 
    for i in range(len(outputs)):
        pro = []
        for x in range(len(targets)):
            iouValue = bbox_iou(outputs[i].t(), targets[x])
            pro.append(iouValue)
        profit_matrix.append(pro)
    match_res = matching2(profit_matrix) # Do Maximum matching

    # Check match result, if matched pair iou < thresold, tell apart them
    correct = 0.
    total = 0.
    match_res_final_cor = []
    match_res_final_err = []
    for i in range(len(match_res)):
        iouutar = targets[match_res[i][1]]
        iouupred = outputs[match_res[i][0]]
        matchiou = bbox_iou(iouutar.t(), iouupred)
        if matchiou <= 0.30:
        #if matchiou <= 1.10:
            match_res_final_err.append((match_res[i][0], match_res[i][0]))
            match_res_final_err.append((match_res[i][1], match_res[i][1]))
        else:
            match_res_final_cor.append((match_res[i][0], match_res[i][1]))

    m0 = []
    m1 = []
    remainin = []
    remaintar = []
    if showid:
        for i in match_res: 
            m0.append(i[0])
            m1.append(i[1])
        for i in range(0, len(inputid)):
            if inputid[i] not in m0: remainin.append(inputid[i])
        for i in range(0, len(tarid)):
            if tarid[i] not in m1: remaintar.append(tarid[i])
    return match_res_final_cor, match_res_final_err, remainin, remaintar