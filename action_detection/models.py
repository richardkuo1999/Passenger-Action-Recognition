import os

import torch.nn.functional as F

from utils.parse_config import *
from utils.utils import *

import torchvision.models as models
#from mm.resnet_new import pose_resnet50
#from mm.resnet_1x1 import pose_resnet50
from mm.seresnet import pose_seresnet50
#from mm.resnet_ch import pose_resnet18
#from mm.resnet_nopretrain import resnet50

ONNX_EXPORT = False

use_feature_map = 60

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    yolo_index = -1

    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                # modules.add_module('leaky_%d' % i, nn.PReLU(num_parameters=filters, init=0.10))
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1, inplace=True))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            yolo_index += 1
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            nc = int(module_def['classes'])  # number of classes
            img_size = hyperparams['height']
            # Define detection layer
            modules.add_module('yolo_%d' % i, YOLOLayer(anchors, nc, img_size, yolo_index))

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    #print(module_list)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

        if ONNX_EXPORT:  # grids must be computed in __init__
            stride = [32, 16, 8][yolo_index]  # stride of this layer
            nx = int(img_size[1] / stride)  # number x grid points
            ny = int(img_size[0] / stride)  # number y grid points
            create_grids(self, max(img_size), (nx, ny))

    def forward(self, p, img_size, var=None):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device)
        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Constants CAN NOT BE BROADCAST, ensure correct shape!
            ngu = self.ng.repeat((1, self.na * self.nx * self.ny, 1))
            grid_xy = self.grid_xy.repeat((1, self.na, 1, 1, 1)).view((1, -1, 2))
            anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view((1, -1, 2)) / ngu

            # p = p.view(-1, 5 + self.nc)
            # xy = torch.sigmoid(p[..., 0:2]) + grid_xy[0]  # x, y
            # wh = torch.exp(p[..., 2:4]) * anchor_wh[0]  # width, height
            # p_conf = torch.sigmoid(p[:, 4:5])  # Conf
            # p_cls = F.softmax(p[:, 5:85], 1) * p_conf  # SSD-like conf
            # return torch.cat((xy / ngu[0], wh, p_conf, p_cls), 1).t()

            p = p.view(1, -1, 5 + self.nc)
            xy = torch.sigmoid(p[..., 0:2]) + grid_xy  # x, y
            wh = torch.exp(p[..., 2:4]) * anchor_wh  # width, height
            p_conf = torch.sigmoid(p[..., 4:5])  # Conf
            p_cls = p[..., 5:5 + self.nc]
            # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
            # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
            p_cls = torch.exp(p_cls).permute((2, 1, 0))
            p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
            p_cls = p_cls.permute(2, 1, 0)
            return torch.cat((xy / ngu, wh, p_conf, p_cls), 2).squeeze().t()

        else:  # inference
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
            # io[..., 5:] = F.softmax(io[..., 5:], dim=4)  # p_cls
            io[..., :4] *= self.stride
            if self.nc == 1:
                io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235


            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 5 + self.nc), p

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg, img_size=(512, 512)):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_defs[0]['cfg'] = cfg
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, x, get_feature_map=None, var=None):
        img_size = max(x.shape[-2:])
        layer_outputs = []
        output = []
        self.get_feature_map = get_feature_map
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                x = module[0](x, img_size)
                output.append(x)
            layer_outputs.append(x)

        '''
        elif ONNX_EXPORT:
            output = torch.cat(output, 1)  # cat 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            nc = self.module_list[self.yolo_layers[0]][0].nc  # number of classes
            return output[5:5 + nc].t(), output[:4].t()  # ONNX scores, boxes
        '''
        if self.training:
            if self.get_feature_map:
                #io, p = list(zip(*output))
                #print('\nget_fea:')
                #print(np.shape(layer_outputs[get_feature_map]))

                # ori code, can run
                return output, layer_outputs[get_feature_map]
                #
                #io, p = list(zip(*output))
                #return torch.cat(io, 1), p, layer_outputs[get_feature_map]
            else:
                return output
        else:
            if self.get_feature_map:
                io, p = list(zip(*output))  # inference output, training output
                return torch.cat(io, 1), p, layer_outputs[get_feature_map]
            else:
                io, p = list(zip(*output))  # inference output, training output
                return torch.cat(io, 1), p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            for i, b in enumerate(a):
                if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                    # fuse this bn layer with the previous conv2d layer
                    conv = a[i - 1]
                    fused = torch_utils.fuse_conv_and_bn(conv, b)
                    a = nn.Sequential(fused, *list(a.children())[i + 1:])
                    break
            fused_list.append(a)
        self.module_list = fused_list
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers

#net1 = models.resnet18(pretrained=True)
class Res(nn.Module):
    def __init__(self):
        super(Res, self).__init__()
        #self.net1 = models.resnet18(pretrained=True)
        self.resnett = pose_seresnet50()
        #self.resnett = pose_resnet50()
        #self.resnett = resnet50()

    def forward(self, x):
        x = self.resnett(x)

        return x

class DarknetRes(nn.Module):
    def __init__(self, cfg_path=None, cfg_path2=None, img_size=512):
        super(DarknetRes, self).__init__()
        self.img_size = img_size
        self.model1 = Darknet(cfg_path, self.img_size)
        self.model2 = Res()

    def forward(self, img, targets=None, conf_thres1=0.5, nms_thres1=0.5):

        #print('\ninput size:')
        #print(np.shape(img))
        # for training or testing
        if targets is not None:
            #inf_out1, train_out1, inf_out2 = self.forward_train(img, targets)
            #print("TARGET IS NOT NONE")

            if self.training:  # for training
                train_out1, inf_out2 = self.forward_train(img, targets)
                return train_out1, inf_out2
            else:              # for testing
                #print("TESTING")
                inf_out1, train_out1, inf_out2 = self.forward_train(img, targets)
                return inf_out1, train_out1, inf_out2

        # for inferencing
        elif targets is None:
            det1, cla_res, inf_out = self.forward_inference(img, conf_thres1, nms_thres1)
            return det1, cla_res, inf_out
        else:
            print('targets: ', targets is None)
            print('LS_targetsis: ', LS_targetsis is None)
            print('error')

    def forward_train(self, img, targets=None):
        # run model1
        if self.training:
            output, feature_map1 = self.model1(img, get_feature_map=use_feature_map)
            #feature_map1 = con33(feature_map1.cuda())
            #inf_out1, train_out1, feature_map1 = self.model1(img, get_feature_map=use_feature_map)
            #print(type(feature_map1))
            #print('\ngot feature map size:')
            #print(np.shape(feature_map1))
        else:
            inf_out1, train_out1, feature_map1 = self.model1(img, get_feature_map=use_feature_map)
        self.nG = feature_map1.shape[-1]
        # after feature map cropped, need to rezie feature map
        self.img_size2 = self.nG // 2
        
        # generate crop_xyxy
        crop_xyxy = targets.clone()
        x, y = targets[:, 2], targets[:, 3]
        w, h = targets[:, 4], targets[:, 5]
        crop_xyxy[:, 2], crop_xyxy[:, 3] = x - (w / 2), y - (h / 2)
        crop_xyxy[:, 4], crop_xyxy[:, 5] = x + (w / 2), y + (h / 2)
        crop_xyxy[:, 2:] *= self.nG

        # limit crop_xyxy
        crop_xyxy = self.xyxy_limit(crop_xyxy)

        '''
        print('\nfeature map 1 original:')
        print(np.shape(feature_map1))
        print('\nimg-size2:')
        print(self.img_size2)
        '''
        #print('\ncrop xyxy:')
        #print(crop_xyxy)
        # Run model2
        feature_map2 = self.crop_roi(feature_map1, crop_xyxy, self.img_size2)

        #print('\nfeature map 2 cropped:')
        #print(np.shape(feature_map2))

        inf_out2 = self.model2(feature_map2)

        #print('\nin models1:')
        #print(inf_out2)
        #print('\ninf out 2:')
        #print(np.shape(inf_out2))

        if self.training:
            return output, inf_out2
        else:
            return inf_out1, train_out1, inf_out2

    def forward_inference(self, img, conf_thres1=0.5, nms_thres1=0.5):
        # run model1
        inf_out1, train_out1, feature_map1 = self.model1(img, get_feature_map=use_feature_map)
        self.nG = feature_map1.shape[-1]
        # after feature map cropped, need to rezie feature map
        self.img_size2 = self.nG // 2
        self.stride = 16

        det1 = non_max_suppression(inf_out1, conf_thres1, nms_thres1)
        crop_xyxy = []
        for i, det in enumerate(det1):
            if det is not None:
                image = i * torch.ones(len(det), 1).to(det.device)
                cls = det[:, 6].clone().unsqueeze(dim=1)
                xyxy = det[:, 0: 4].clone() / self.stride
                #xyxy = det[:, 0: 4].clone()
                crop_xyxy.append(torch.cat((image, cls, xyxy), 1))
        if len(crop_xyxy): 
            crop_xyxy = torch.cat(crop_xyxy, 0)
        else:              
            return det1, None

        #print('----------------')
        #print('\nlimited crop1:')
        #print(crop_xyxy)

        # limit crop_xyxy
        #print(type(crop_xyxy))
        crop_xyxy = self.xyxy_limit(crop_xyxy)

        #print('\nlimited crop2:')
        #print(crop_xyxy)
        #print(np.shape(feature_map1))

        # run model2
        feature_map2 = self.crop_roi(feature_map1, crop_xyxy, self.img_size2)

        inf_out2 = self.model2(feature_map2)

        #print('\nin models2:')
        #print(inf_out2)

        # predict to class
        _, cla_res = inf_out2.max(1)
        #print('\nin models:')
        #print(cla_res)
        #print('\ndet1:')
        #print(det1)
        return det1, cla_res, inf_out2 

    def xyxy_limit(self, crop_xyxy):
        crop_xyxy = crop_xyxy.round()

        # if grid x,y equal to the size of cropped feature map
        # left up point
        crop_xyxy[crop_xyxy[:, 2] == self.nG, 2] -= 1
        crop_xyxy[crop_xyxy[:, 3] == self.nG, 3] -= 1
        # right down point
        crop_xyxy[crop_xyxy[:, 2] == crop_xyxy[:, 4], 4] += 1
        crop_xyxy[crop_xyxy[:, 3] == crop_xyxy[:, 5], 5] += 1

        #print(crop_xyxy[:, 3] == crop_xyxy[:, 5])

        # if grid x,y smaller than 0
        # left up point
        crop_xyxy[crop_xyxy[:, 2] < 0, 2] = 0
        crop_xyxy[crop_xyxy[:, 3] < 0, 3] = 0
        # right down point
        crop_xyxy[crop_xyxy[:, 4] > self.nG, 4] = self.nG - 1
        crop_xyxy[crop_xyxy[:, 5] > self.nG, 5] = self.nG- 1

        crop_xyxy[crop_xyxy[:, 2] >= self.nG, 2] = self.nG - 2
        crop_xyxy[crop_xyxy[:, 3] >= self.nG, 3] = self.nG- 2

        crop_xyxy[crop_xyxy[:, 2] == crop_xyxy[:, 4], 4] += 1
        crop_xyxy[crop_xyxy[:, 3] == crop_xyxy[:, 5], 5] += 1

        #crop_xyxy = torch.tensor([[0,0,4,12,15,17],[0,0,4,17,13,23],[0,0,19,6,29,12],[0,0,13,4,23,13]])
        #crop_xyxy = [[0,0,4,12,15,17],[0,0,4,17,13,23],[0,0,19,6,29,12],[0,0,13,4,23,13]]

        crop_xyxy = crop_xyxy.cpu().numpy().astype(int)
        return crop_xyxy

    def crop_roi(self, feature_map, crop_xyxy, img_size):
        new_feature_map = []
        for i, image in enumerate(crop_xyxy[:, 0]):
            #print('\ni&image:')
            #print(i,image)
            #print(img_size)
            #print(np.shape(feature_map))

            #print('\n(h1,h2):')
            #print(crop_xyxy[i, 3], crop_xyxy[i, 5])
            #print('\n(w1,w2):')
            #print(crop_xyxy[i, 2], crop_xyxy[i, 4])
            # crop from feature map
            #print('\nfeature map:')
            #print(np.shape(feature_map))

            crop_feature = feature_map[image, :, crop_xyxy[i, 3]: crop_xyxy[i, 5], crop_xyxy[i, 2]: crop_xyxy[i, 4]].clone()  # b, c, h, w
            #crop_feature = feature_map[image, :, crop_xyxy[i, 2]: crop_xyxy[i, 4], crop_xyxy[i, 3]: crop_xyxy[i, 5]].clone()  # b, c, h, w

            #print('\ncrop feature_map:')
            #print(np.shape(crop_feature))
            #print('----------------')
            crop_feature = crop_feature.unsqueeze(dim=0)
            # resize to unified size
            crop_feature = F.interpolate(crop_feature, size=(img_size, img_size), mode='nearest')
            new_feature_map.append(crop_feature)
        new_feature_map = torch.cat(new_feature_map, 0)
        return new_feature_map

def get_yolo_layers(model):
    a = [module_def['type'] == 'yolo' for module_def in model.module_defs]
    return [i for i, x in enumerate(a) if x]  # [82, 94, 106] for yolov3


def create_grids(self, img_size=416, ng=(13, 13), device='cpu'):
    nx, ny = ng  # x and y grid size
    self.img_size = img_size
    self.stride = img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).float().view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            url = 'https://pjreddie.com/media/files/' + weights_file
            print('Downloading ' + url)
            os.system('curl ' + url + ' -o ' + weights)
        except IOError:
            print(weights + ' not found.\nTry https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return cutoff


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')
