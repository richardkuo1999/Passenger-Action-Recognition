import csv
import numpy as np
import torch
import random
import torch_utils

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    #print(n_correct_elems / batch_size)
    return n_correct_elems / batch_size

# Add function to get each classes accuracy
def calculate_accuracy_for_test(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred2 = pred.clone()
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    correct_sitting, correct_standing, n_sitting, n_standing = 0., 0., 0., 0.
    correct_sit, correct_stand, n_sit, n_stand = 0., 0., 0., 0.
    for c in range(targets.size(0)):
        if targets[c]==0:
            if pred2[c].eq(targets[c])==1:
                correct_sitting = correct_sitting+1
                n_sitting = n_sitting+1
            else:
                n_sitting = n_sitting+1
        if targets[c]==1:
            if pred2[c].eq(targets[c].to(dtype=torch.int64))==1:
                correct_standing = correct_standing+1
                n_standing = n_standing+1
            else:
                n_standing = n_standing+1
        if targets[c]==2:
            if pred2[c].eq(targets[c])==1:
                correct_sit = correct_sit+1
                n_sit = n_sit+1
            else:
                n_sit = n_sit+1
        if targets[c]==3:
            #print('\ntarget 4!')
            if pred2[c].eq(targets[c].to(dtype=torch.int64))==1:
                correct_stand = correct_stand+1
                n_stand = n_stand+1
            else:
                n_stand = n_stand+1
    '''
    print('in cal_acc:')
    print(n_sitting, n_standing, correct_sitting, correct_standing)
    '''
    return n_correct_elems / batch_size, n_sitting, n_standing, correct_sitting, correct_standing, n_sit, n_stand,\
    correct_sit, correct_stand
