import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torchvision.models as models
import numpy as np
import pretrainedmodels

class Net(torch.nn.Module):
    def __init__(self , modelx):
        super(Net, self).__init__()
        #self.conv1 = torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        #self.conv2 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        #self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, bias=False)
        #self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=3, bias=False)
        # for resnet18
        #self.resnet_layer = torch.nn.Sequential(*list(modelx.children())[6:-1])
        # for resnet50
        # 7: 1024ch 73, 6: 512ch 60, 5: 256ch 35
        self.seresnet_layer = torch.nn.Sequential(*list(modelx.children())[6:-1])

        #self.resnet_layer = torch.nn.Sequential(*list(modelx.modules())[31:-1])
        self.Linear_layer = torch.nn.Linear(131072, 2)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        #print('\ninput to cla:')
        #print(np.shape(x))
        #print(np.shape(x))
        x = self.seresnet_layer(x)
        x = torch.flatten(x, 1)
        x = self.Linear_layer(x)
        return x

def pose_seresnet50(pretrained=False, progress=True, **kwargs):
    net1 = pretrainedmodels.__dict__['se_resnext50'](num_classes=1000, pretrained='imagenet')
    output = Net(net1)
    return output

if __name__ == '__main__':

    net = pretrainedmodels.__dict__['se_resnext50'](num_classes=1000, pretrained='imagenet')
    #model = pose_resnet50()
    print(net)