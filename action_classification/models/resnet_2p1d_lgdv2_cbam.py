import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_inplanes():
    return [64, 128, 256, 512]

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


def conv1x3x3(in_planes, mid_planes, stride=1):
    return nn.Conv3d(in_planes,
                     mid_planes,
                     kernel_size=(1, 3, 3),
                     stride=(1, stride, stride),
                     padding=(0, 1, 1),
                     bias=False)


def conv3x1x1(mid_planes, planes, stride=1):
    return nn.Conv3d(mid_planes,
                     planes,
                     kernel_size=(3, 1, 1),
                     stride=(stride, 1, 1),
                     padding=(1, 0, 0),
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1
        self.conv1_s = conv1x3x3(in_planes, mid_planes1, stride)
        self.bn1_s = nn.BatchNorm3d(mid_planes1)
        self.conv1_t = conv3x1x1(mid_planes1, planes, stride)
        self.bn1_t = nn.BatchNorm3d(planes)

        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(planes, mid_planes2)
        self.bn2_s = nn.BatchNorm3d(mid_planes2)
        self.conv2_t = conv3x1x1(mid_planes2, planes)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)

        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2p1d_parameters = planes * 3 * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv2_s = conv1x3x3(planes, mid_planes, stride)
        self.bn2_s = nn.BatchNorm3d(mid_planes)
        self.conv2_t = conv3x1x1(mid_planes, planes, stride)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        #Implement LGD block
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(planes * 4 // 2, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        #self.fc2 = nn.Conv3d(planes * 4 // 16, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm3d(planes * 4)
        self.fc3 = nn.Conv3d(planes * 4, planes * 4 // 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc4 = nn.Conv3d(planes * 4 // 16, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc5 = nn.Conv3d(planes * 4, planes * 4 // 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc6 = nn.Conv3d(planes * 4 // 16, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace=True)

        #Implement SpatialAttention block
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv4 = nn.Conv3d(2, 1, 3, padding=1, bias=False)
        #self.conv4 = nn.Conv3d(planes * 4, planes * 4 // 16, 1, padding=0, bias=False)
        #self.conv5 = nn.Conv3d(planes * 4 // 16, planes * 4, 1, padding=0, bias=False)
        #self.bn5 = nn.BatchNorm3d(1)
        #self.conv4 = nn.Conv3d(1, 1, 3, padding=1, bias=False)
        
    def forward(self, xx):
        # xx contains two element: input->x and global path->glo
        x = xx[0]
        glo = xx[1]
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #out = self.relu(out)

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

            # Implement SpatialAttention block
            avg_out = torch.mean(out, dim=1, keepdim=True)
            max_out, _ = torch.max(out, dim=1, keepdim=True)
            sa = torch.cat([avg_out, max_out], dim=1)
            #sa = max_out
            sa = self.conv4(sa)
            #sa = self.conv4(out)
            #sa = self.bn5(sa)
            #sa = self.relu(sa)
            #sa = self.conv5(sa)            
            sa = self.sigmoid(sa)
            out = out * sa
            #out = self.relu(out)

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
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 num_classes=400,
                 sample_size=112,
                 sample_duration=15):
        #super().__init__()
        super(ResNet, self).__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv1_s = nn.Conv3d(n_input_channels,
                                 mid_planes,
                                 kernel_size=(1, 7, 7),
                                 stride=(1, 2, 2),
                                 padding=(0, 3, 3),
                                 bias=False)
        self.bn1_s = nn.BatchNorm3d(mid_planes)
        self.conv1_t = nn.Conv3d(mid_planes,
                                 self.in_planes,
                                 kernel_size=(conv1_t_size, 1, 1),
                                 stride=(conv1_t_stride, 1, 1),
                                 padding=(conv1_t_size // 2, 0, 0),
                                 bias=False)
        self.bn1_t = nn.BatchNorm3d(self.in_planes)
        #self.relu = nn.ReLU6(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
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
        if lookshape:
            print('loc & g:--------')
            print(np.shape(loc))
            print(np.shape(g))
            print('----------------')
        x = self.avgpool(loc)
        if lookshape:
            print('\nlayer5-------------')
            print(np.shape(x))
            print('--------------')
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    model_info(model,'full')
    return model 

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model