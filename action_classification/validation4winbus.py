import torch
from torch.autograd import Variable
import time
import sys
from apex import amp
#from utils import AverageMeter, calculate_accuracy
from utils import AverageMeter, calculate_accuracy_for_test
import time
import os
import numpy as np
from torchvision.utils import save_image

def val_epoch(epoch, data_loader, model, criterion, opt, logger, best_acc, optimizer=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    num_sitting, num_standing, correct_sitting, correct_standing = 0., 0., 0., 0.,
    num_sit, num_stand, correct_sit, correct_stand = 0., 0., 0., 0.,
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        #data_time = 0
        if not opt.no_cuda:
            #targets = targets.cuda(async=True)
            targets = targets.cuda()
        st = time.time()
        inputs = Variable(inputs, volatile=True).cuda()
        targets = Variable(targets, volatile=True).cuda()
        outputs = model(inputs)
        print('time: %.5f'%(time.time() - st))
        loss = criterion(outputs, targets)
        #acc = calculate_accuracy(outputs, targets)
        
        # n_<action>: number of the action in one batch
        # c_<action>: number of correctly classified action in one batch
        acc, n_sitting, n_standing, c_sitting, c_standing, n_sit, n_stand, c_sit, c_stand = calculate_accuracy_for_test(outputs, targets)

        num_sitting = num_sitting + n_sitting
        num_standing = num_standing + n_standing
        correct_sitting = correct_sitting + c_sitting
        correct_standing = correct_standing + c_standing
        num_sit = num_sit + n_sit
        num_stand = num_stand + n_stand
        correct_sit = correct_sit + c_sit
        correct_stand = correct_stand + c_stand

        #losses.update(loss.data[0], inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))
    print(correct_sitting, correct_standing, correct_sit, correct_stand)
    logger.log({'epoch': epoch, 'loss': losses.avg,\
     'acc': accuracies.avg,\
     'acc_sitting': correct_sitting/num_sitting,\
     'acc_standing': correct_standing/num_standing,\
     'acc_sit': correct_sit/num_sit,\
     'acc_stand': correct_stand/num_stand,\
     'time': time.time()})
    print('in val epoch:')
    print('acc_sitting:',correct_sitting/num_sitting, ' acc_standing:',correct_standing/num_standing)
    print('acc_sit:',correct_sit/num_sit, ' acc_stand:',correct_stand/num_stand)

    acc = round(accuracies.avg,4)
    '''
    if not opt.no_train:
        if acc > best_acc:
            print('acc: ', acc, ' best_acc: ', best_acc)
            print(time.time())
            save_best_path = os.path.join(opt.result_path,'best.pth')
            states = {
                'epoch': epoch,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_best_path)
    '''
    return losses.avg, acc