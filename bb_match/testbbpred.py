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

def bbox_iou(box1, box2):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box1 = box1.t()
    box2 = box2.t()
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    #print('b1_x1:',np.shape(b1_x1))
    #print((torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0))

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    #print('inter:',np.shape(inter))

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c_area = cw * ch + 1e-16  # convex area
    return iou - (c_area - union) / c_area, iou  # GIoU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net6().to(device)
checkpoint = torch.load('best-new.pth')
net.load_state_dict(checkpoint['state_dict'])
net.eval()

box1 = torch.tensor([[0.8389,0.2056,0.3229,0.4099],
                     [0.5109,0.1072,0.1705,0.2132]]).to(device).float()
box2 = torch.tensor([[0.1849,0.1385,0.2936,0.2757],
                     [0.4865,0.2217,0.2408,0.4421]]).to(device).float()

#box3 = torch.tensor([[0.5109,0.1072,0.1705,0.2132]]).to(device).float() 
#box4 = torch.tensor([[0.4865,0.2217,0.2408,0.4421]]).to(device).float()

output = net(box1)
print(output)
#output[:,:2] = output.sigmoid()
#output[:,2:4] = output[:,2:4].exp().clamp(max=1E3)
giou, iou = bbox_iou(output, box2)
print(iou)
