import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]

def look_bottleneck_global(glo):
    if look_bottleneck_global:
        if glo is None:
            print('first bottleneck-> no global content!')
        else:
            print('glo has content!')

# Can print the model structure
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

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

# Implement of bottleneck with se block
class BottleneckX(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, first_block=False):
        super(BottleneckX, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        #self.bn1 = nn.GroupNorm(4, planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        #self.bn2 = nn.GroupNorm(4, planes)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        #self.bn3 = nn.GroupNorm(4, planes * 4)

        self.downsample = downsample
        self.stride = stride
        # If first bottleneckX, it does not contain global path
        self.first_block = first_block
        # If downsampling occurs, set true
        self.ds = False
        #self.se_module = SEModule(planes * 4, reduction=16, first_block=self.first_block)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        #Implement LGD block
        self.fc1 = nn.Conv3d(planes * 4 // 2, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        #self.fc2 = nn.Conv3d(planes * 4 // 16, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm3d(planes * 4)
        #self.bn4 = nn.GroupNorm(4, planes * 4)

        self.fc3 = nn.Conv3d(planes * 4, planes * 4 // 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc4 = nn.Conv3d(planes * 4 // 16, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)

        self.fc5 = nn.Conv3d(planes * 4, planes * 4 // 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc6 = nn.Conv3d(planes * 4 // 16, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace=True)

        #Implement SpatialAttention block
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv4 = nn.Conv3d(2, 1, 3, padding=1, bias=False)

    def forward(self, xx):
        # xx contains two element: input->x and global path->glo
        x = xx[0]
        glo = xx[1]
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #out = self.relu(out)

        # If downsample, downsampleing global path & residual channels
        if self.downsample is not None:
            if glo is not None:
                glo = self.avg_pool(glo)
                glo = self.fc1(glo)
                glo = self.relu(glo)
            residual = self.downsample(x)
        #LGD block
        if glo is not None:
            glo = self.fc3(glo)
            glo = self.relu(glo)
            glo = self.fc4(glo)
            glo = self.sigmoid(glo)

            out = out * glo
            #out = self.relu(out)

            # Implement SpatialAttention block
            avg_out = torch.mean(out, dim=1, keepdim=True)
            max_out, _ = torch.max(out, dim=1, keepdim=True)
            sa = torch.cat([avg_out, max_out], dim=1)
            sa = self.conv4(sa)            
            sa = self.sigmoid(sa)
            out = out * sa

            glo2 = self.avg_pool(out)
            glo2 = self.fc5(glo2)
            glo2 = self.relu(glo2)
            glo2 = self.fc6(glo2)
            glo2 = self.sigmoid(glo2)

            g = glo + glo2
            g = self.relu(g)
            out = out + residual
            out = self.relu(out)
            outg = [out, g]
        # Normal bottleneck
        else:
            out = out + residual
            out = self.relu(out)
            outg = [out, residual]

        return outg


class ResNet(nn.Module):

    def __init__(self,
                 blockx,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        #self.bn1 = nn.GroupNorm(4, 64)

        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(blockx, 64, layers[0], shortcut_type, first_block=True)
        self.layer2 = self._make_layer(blockx, 128, layers[1], shortcut_type, stride=2, first_block=False)
        self.layer3 = self._make_layer(blockx, 256, layers[2], shortcut_type, stride=2, first_block=False)
        self.layer4 = self._make_layer(blockx, 512, layers[3], shortcut_type, stride=2, first_block=False)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        #last_size = 4
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * blockx.expansion, num_classes)
        #self.fusion = nn.Conv3d(512 * block.expansion * 2, 512 * block.expansion, kernel_size=1, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, first_block=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, first_block))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print('lgd')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        lookshape = False
        # First time need to give two element to model
        xx = [x, None]
        x = self.layer1(xx)
        if lookshape:
            print('\nlayer1-------------')
            print(np.shape(x[0]))
            print(np.shape(x[1]))
            print('--------------')

        x = self.layer2(x)
        if lookshape:
            print('\nlayer2-------------')
            print(np.shape(x[0]))
            print(np.shape(x[1]))
            print('--------------')

        x = self.layer3(x)
        if lookshape:
            print('\nlayer3-------------')
            print(np.shape(x[0]))
            print(np.shape(x[1]))
            print('--------------')

        x = self.layer4(x)
        if lookshape:
            print('\nlayer4-------------')
            print(np.shape(x[0]))
            print(np.shape(x[1]))
            print('--------------')

        # After bottlenck part
        loc, g = x[0], x[1]
        #print(g)
        if lookshape:
            print('loc & g:--------')
            print(np.shape(loc))
            print(np.shape(g))
            print('----------------')


        x = self.avgpool(loc)

        #x = x + g
        #x = self.bn2(x)
        #x = self.relu(x)

        if lookshape:
            print('\nlayer5-------------')
            print(np.shape(x))
            print('--------------')

        # Test local and global path feature maps fusion type below
        
        # 3d conv
        #x = torch.cat((x, g), 1)
        #x = self.fusion(x)
        #x = self.bn2(x)
        #x = self.relu(x)

        # concat (need to change fc layer filter number)
        #x = torch.cat((x, g), 1)
        #x = self.relu(x)

        x = x.view(x.size(0), -1)
        if lookshape:
            print('\nlayer6-------------')
            print(np.shape(x))
            print('--------------')

        x = self.fc(x)

        if lookshape:
            print('\nlayer7-------------')
            print(np.shape(x))
            print('--------------')

        return x

def get_fine_tuning_parameters(model, ft_begin_index):
    #if ft_begin_index == 0:
    #    return model.parameters()
    print('ohraaaa')
    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    # Look the content of ft_module
    print('ft: ', ft_module_names)

    parameters = []
    ii = 0

    '''
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ii >= 271: #220 271
                print(ii)
                parameters.append({'params': v})
            else:
                print('notfc')
                print(ii)
                parameters.append({'params': v, 'lr': 0.0})
                #parameters.append({'params': v})
        print(k)
        ii = ii+1
    return parameters
    '''
    
    # bakup code
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
            #if ii >= 271:
                print('fc')
                #print(ii)
                parameters.append({'params': v})
                break
        else:
            print('notfc')
            #print(ii)
            #parameters.append({'params': v, 'lr': 0.0})
            parameters.append({'params': v})
        print(k)
        ii = ii+1
    return parameters
    

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(BottleneckX, [3, 4, 6, 3], **kwargs)
    #model = ResNet(Bottleneck, BottleneckX, [3, 4, 23, 3], **kwargs)
    #model_info(model,'full')
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(BottleneckX, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model