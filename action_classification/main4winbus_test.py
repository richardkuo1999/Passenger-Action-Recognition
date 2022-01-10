import os
import sys
import json
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler


import argparse
from model_c import generate_model
from mean import get_mean, get_std
from spatial_transforms_winbus import (
    Compose, Normalize, RandomHorizontalFlip, ToTensor, RandomVerticalFlip, 
    ColorAugment)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation4winbus import val_epoch
#from validation import val_epoch

import test
from utils import init_seeds
import time

from torchvision import transforms
import PIL
from PIL import Image, ImageOps

# To fixed scale of person
def letterbox(img, resize_size, mode='square'):
    shape = [img.size[1],img.size[0]]  # current shape [height, width]
    new_shape = resize_size
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = img.resize(new_unpad,PIL.Image.ANTIALIAS)
    img = ImageOps.expand(img, border=(left,top,right,bottom), fill=(128,128,128))
    return img

#Override transform's resize function(using letterbox)
class letter_img(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        #letterbox(img,self.size).save('out.bmp')
        return letterbox(img, self.size)
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

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

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, gamma=0.5, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        # If no use of pytorch original crossentropy (use label smooth ce)
        #self.reduction = loss_fcn.reduction
        #self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability
        # If no use of pytorch original crossentropy (use label smooth ce)
        '''
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        '''
        # If use label smooth ce
        return loss.mean()

parser = argparse.ArgumentParser()
parser.add_argument( '--root_path', default='.',
                    type=str, help='Root directory path of data')
parser.add_argument( '--video_path',default='datasets/testdata/winbus-06-2RoHf',
                    type=str, help='Directory path of Videos')
parser.add_argument( '--annotation_path',default='data/winbusTest-06/winbus_06-2-3-4_test2times.json',
                    type=str, help='Annotation file path')
parser.add_argument( '--result_path',default='results/train_results_06',
                    type=str, help='Result directory path')
parser.add_argument( '--dataset',default='ucf101',
                    type=str, help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
parser.add_argument( '--n_classes',default=4,
                    type=int, help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
parser.add_argument( '--n_finetune_classes', default=400,
                    type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
parser.add_argument( '--sample_size', default=112,
                    type=int, help='Height and width of inputs')
parser.add_argument( '--sample_duration',default=16,
                    type=int, help='Temporal duration of inputs')
parser.add_argument( '--initial_scale', default=1.0,
                    type=float, help='Initial scale for multiscale cropping')
parser.add_argument( '--n_scales', default=5,
                    type=int, help='Number of scales for multiscale cropping')
parser.add_argument( '--scale_step', default=0.84089641525,
                    type=float, help='Scale step for multiscale cropping')
parser.add_argument( '--train_crop', default='corner',type=str,
                    help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
parser.add_argument( '--learning_rate', default=0.1,
                    type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
parser.add_argument( '--momentum', default=0.9, 
                    type=float, help='Momentum')
parser.add_argument( '--dampening', default=0.9, 
                    type=float, help='dampening of SGD')
parser.add_argument( '--weight_decay', default=1e-3, 
                    type=float, help='Weight Decay')
parser.add_argument( '--mean_dataset', default='activitynet',
                    type=str, help='dataset for mean values of mean subtraction (activitynet | kinetics)')
parser.add_argument( '--no_mean_norm', default=False,action='store_true',
                        help='If true, inputs are not normalized by mean.')
parser.add_argument( '--std_norm', default=False, action='store_true',
                    help='If true, inputs are normalized by standard deviation.')
parser.add_argument('--nesterov', default=False, action='store_true',
                    help='Nesterov momentum')
parser.add_argument( '--optimizer', default='sgd',
                    type=str, help='Currently only support SGD')
parser.add_argument( '--lr_patience', default=10,type=int, 
                    help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
parser.add_argument( '--batch_size', default=16, 
                    type=int, help='Batch Size')
parser.add_argument( '--n_epochs', default=81,
                    type=int, help='Number of total epochs to run')
parser.add_argument( '--begin_epoch', default=1, type=int,
                    help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
parser.add_argument( '--n_val_samples', default=1, type=int,
                    help='Number of validation samples for each activity')
parser.add_argument( '--resume_path', default='results/train_results_06/best.pth', type=str,
                    help='Save data (.pth) of previous training')
parser.add_argument( '--pretrain_path', default='', 
                    type=str, help='Pretrained model (.pth)')
parser.add_argument( '--ft_begin_index', default=0, 
                    type=int, help='Begin block index of fine-tuning')
parser.add_argument( '--no_train', default=True, action='store_true',
                    help='If true, training is not performed.')
parser.add_argument( '--no_val', default=True, action='store_true',
                    help='If true, validation is not performed.')
parser.add_argument( '--test', default=True, action='store_true',
                    help='If true, test is performed.')
parser.add_argument( '--test_subset', default='val',
                    type=str, help='Used subset in test (val | test)')
parser.add_argument( '--scale_in_test', default=1.0,
                    type=float, help='Spatial scale in test')
parser.add_argument( '--crop_position_in_test', default='c',
                    type=str, help='Cropping method (c | tl | tr | bl | br) in test')
parser.add_argument( '--no_softmax_in_test', default=False, action='store_true',
                    help='If true, output for each clip is not normalized using softmax.')
parser.add_argument('--no_cuda', default=False, action='store_true', 
                    help='If true, cuda is not used.')
parser.add_argument( '--n_threads', default=4,
                    type=int, help='Number of threads for multi-thread loading')
parser.add_argument( '--checkpoint', default=10,
                    type=int, help='Trained model is saved at every this epochs.')
parser.add_argument( '--no_hflip', default=False, action='store_true',
                    help='If true holizontal flipping is not performed.')
parser.add_argument( '--norm_value', default=1, type=int,
                    help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
parser.add_argument( '--model', default='resnet', type=str,
                    help='(resnet | preresnet | wideresnet | resnext | densenet | ')
parser.add_argument( '--model_depth', default=50, type=int,
                    help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
parser.add_argument( '--resnet_shortcut', default='B', type=str, 
                    help='Shortcut type of resnet (A | B)')
parser.add_argument( '--wide_resnet_k', default=2, 
                    type=int, help='Wide resnet k')
parser.add_argument( '--resnext_cardinality', default=32,
                    type=int, help='ResNeXt cardinality')
parser.add_argument( '--manual_seed', default=1, 
                    type=int, help='Manually set random seed')
parser.add_argument( '--use_mix', action='store_true',
                    help='If true, use apex.')
parser.add_argument( '--test_weight', default=True, action='store_true',
                    help='If true, can test the trained model.')

opt = parser.parse_args()

if __name__ == '__main__':
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # Ori code
    torch.manual_seed(opt.manual_seed)

    # Initialize
    #init_seeds()
    #torch.manual_seed(opt.manual_seed)
    #torch.cuda.manual_seed(opt.manual_seed)
    #torch.backends.cudnn.deterministic = True
    np.random.seed(opt.manual_seed)
    random.seed(opt.manual_seed)

    if not opt.use_mix:
        model, parameters = generate_model(opt)
    else:
        model, parameters, optimizer, scheduler = generate_model(opt)

    #criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()
    #criterion = LabelSmoothingCrossEntropy()
    g = 2
    criterion = FocalLoss(criterion, g)

    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        print('mean:', opt.mean)
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        
        spatial_transform = Compose([
            letter_img(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        
        '''
        #Pytorch build-in function(preprocess image)
        spatial_transform = transforms.Compose([
            letter_img(112),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])
        '''
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        #temporal_transform = TemporalCenterCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            #batch_size=1,
            shuffle=True,
            num_workers=opt.n_threads,
            #num_workers=1,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        
        if not opt.use_mix:
            if opt.nesterov:
                dampening = 0
            else:
                dampening = opt.dampening
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
            #tmax = opt.n_epochs//2 + 1
            tmax = opt.n_epochs + 1
            #scheduler = lr_scheduler.ReduceLROnPlateau(
            #    optimizer, 'min', patience=opt.lr_patience, verbose=True)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = tmax)

    if not opt.no_val:
        print('norm:', opt.norm_value)

        spatial_transform = Compose([
            #Scale(opt.sample_size),
            #CenterCrop(opt.sample_size),
            letter_img(opt.sample_size),
            ToTensor(opt.norm_value), 
            norm_method
        ])
        
        '''
        #Pytorch build-in function(preprocess image)
        spatial_transform = transforms.Compose([
            letter_img(opt.sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])
        '''
        #temporal_transform = LoopPadding(opt.sample_duration)
        #temporal_transform = TemporalRandomCrop(opt.sample_duration)
        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            #batch_size=opt.batch_size,
            batch_size=1,
            shuffle=True,
            #shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc', 'acc_sitting', 'acc_standing',\
            'acc_sit', 'acc_stand', 'time'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        
        if not opt.test_weight:
            #changed
            model_dict = model.state_dict()
            pretrain_dict = {k:v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(model_dict)
        else:
            model = nn.DataParallel(model, device_ids=None)
            model.load_state_dict(checkpoint['state_dict'])

        #if not opt.no_train:
        #    optimizer.load_state_dict(checkpoint['optimizer'])

    #model_info(model,'full')
    #tmax = opt.n_epochs//2 + 1
    print('run')
    best_acc = 0.
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            if not opt.no_train:
                validation_loss, acc = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger, best_acc)
            else:
                validation_loss, acc = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger, best_acc)

        # Save the best accuracy weight
        if not opt.no_train:
            if acc >= best_acc:
                print('acc: ', acc, ' best_acc: ', best_acc)
                print(time.time())
                save_best_path = os.path.join(opt.result_path,'best.pth')
                states = {
                    'epoch': i + 1,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_best_path)
                best_acc = acc
                del states

        if not opt.no_train and not opt.no_val:
            #scheduler.step(validation_loss)
            scheduler.step()
            #if (i + 1) % tmax==0:
            #    for _ in range(tmax):
            #        scheduler.step()

    if opt.test:
        spatial_transform = Compose([
            #Scale(int(opt.sample_size / opt.scale_in_test)),
            #Scale(opt.sample_size),
            #CornerCrop(opt.sample_size, opt.crop_position_in_test),
            #CenterCrop(opt.sample_size),
            letter_img(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        
        '''
        spatial_transform = transforms.Compose([
            letter_img(112),
            transforms.ToTensor(),
            #transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])
        '''
        #temporal_transform = LoopPadding(opt.sample_duration)
        #temporal_transform = TemporalRandomCrop(opt.sample_duration)
        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        target_transform = VideoID()
        
        '''
        # Cal acc
        validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
        '''

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        print(test_data.class_names)
        test.test(test_loader, model, opt, test_data.class_names)
