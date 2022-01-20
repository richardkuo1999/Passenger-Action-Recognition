# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:00:04 2020

@author: Admin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 101)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        #print(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
    
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 512)
        self.fc3 = nn.Linear(512, 2048)
        self.fc4 = nn.Linear(2048, 1001)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

class PredictY(nn.Module):
    def __init__(self):
        super(PredictY, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

class PredictYWH(nn.Module):
    def __init__(self):
        super(PredictYWH, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 100)
        self.fc5 = nn.Linear(512, 100)
        self.fc6 = nn.Linear(512, 100)
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = self.fc3(x)
        out = self.relu(out)
        y = self.fc4(out)
        w = self.fc5(out)
        h = self.fc6(out)
        return y, w, h

class PredictXYWH(nn.Module):
    def __init__(self):
        super(PredictXYWH, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 100)
        self.fc5 = nn.Linear(512, 100)
        self.fc6 = nn.Linear(512, 100)
        self.fc7 = nn.Linear(512, 100)
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = self.fc3(x)
        out = self.relu(out)
        xx = self.fc4(out)
        y = self.fc5(out)
        w = self.fc6(out)
        h = self.fc7(out)
        #print(xx)
        return xx, y, w, h

class PredictXYWH2(nn.Module):
    def __init__(self):
        super(PredictXYWH2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

class PredictXYWH3(nn.Module):
    def __init__(self):
        super(PredictXYWH3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4, 8)
        self.bn1 = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # For predict y
        self.Ypred = PredictY()

    def forward(self, x):
        y = x
        #wh = x[:,1:]
        y = self.Ypred(y)
        #w, h = self.WHpred(wh)
        #return y, w, h
        return y

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # For predict y
        self.YWHpred = PredictYWH()

    def forward(self, x):
        y = x
        #wh = x[:,1:]
        y, w, h = self.YWHpred(y)
        #w, h = self.WHpred(wh)
        #return y, w, h
        return y, w, h

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # For predict y
        self.XYWHpred = PredictXYWH()

    def forward(self, x):
        y = x
        #wh = x[:,1:]
        xx, y, w, h = self.XYWHpred(y)
        #w, h = self.WHpred(wh)
        #return y, w, h
        return xx, y, w, h

class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.XYWHpred = PredictXYWH2()

    def forward(self, x):
        y = x
        xx = self.XYWHpred(y)
        xx = xx.sigmoid()
        return xx

class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.XYWHpred = PredictXYWH3()

    def forward(self, x):
        y = x
        xx = self.XYWHpred(y)
        xx = xx.sigmoid()
        return xx