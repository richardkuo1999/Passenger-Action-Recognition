import glob
import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from . import torch_utils  # , google_utils
'''
#for resnet
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image,ImageOps
import PIL
from torch.autograd import Variable
'''
matplotlib.rc('font', **{'size': 11})

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)

def floatn(x, n=3):  # format floats to n decimals
    return float(format(x, '.%gf' % n))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report is 'full':
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))

def compute_loss(y, w, h, targets):
    criterion = nn.CrossEntropyLoss()
    tarlen = len(targets)
    #print(y)
    loss_y = criterion(y, targets[0:tarlen:3])
    loss_w = criterion(w, targets[1:tarlen:3])
    loss_h = criterion(h, targets[2:tarlen:3])
    #loss = loss_y + 0.05 * (loss_w + loss_h)
    loss = loss_y

    return loss

# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None, person_count=0,pose_class=''):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    '''
    #for resnet
    temp = img[int(x[1]):int(x[2]), int(x[0]):int(x[2])]
    cropped = Image.fromarray(cv2.cvtColor(temp,cv2.COLOR_BGR2RGB))
    cropped_tensor = transform_test(cropped)
    cropped_tensor.unsqueeze_(0)
    cropped_variable = Variable(cropped_tensor)
    classify_result = model(cropped_variable)
    re_class = classify_result.data.numpy().argmax()
    '''
    cv2.circle(img, (int((x[0]+x[2])/2), int((x[1]+x[3])/2)), 5, (0,0,255), 4)
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        #label = pose_class + ',' + str(person_count) + ',' + label
        #label = 'person' + ',' + str(person_count) + ',' + label
        #label = str(person_count) + ',' + label
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_images(imgs, targets, paths=None, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs ** 0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)], fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close()
