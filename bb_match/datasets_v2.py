# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:42:18 2020

@author: Admin
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, batch_size=16):
        self.path = path
        with open(path, 'r') as f:
            img_files = f.read().splitlines()
            ids = []
            files = []
            bbs = []
            for i in range(len(img_files)):
                #ids.append(np.array([float(t) for t in img_files[i].split(' ')[0]]))
                ids.append(np.array([int(img_files[i].split(' ')[0])]))
                files.append(np.array([float(t) for t in img_files[i].split(' ')[1:5]]))
                # For sim data format
                #bbs.append(np.array([float(t) for t in img_files[i].split(' ')[6:]]))
                # For winbus data format
                bbs.append(np.array([float(t) for t in img_files[i].split(' ')[5:]]))
            # Frame id
            self.ids = ids
            # Ground truth
            self.img_files = files
            # Data
            self.bbs = bbs
            
        n = len(self.img_files)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        assert n > 0, 'No images found in %s' % path

        self.n = n
        self.batch = bi  # batch index of image
        
        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.labels = [None] * n

        self.firstread = True
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_path = self.img_files[index]

        # Load image
        img = self.imgs[index]
        if img is None:
            #img = self.bbs[index]
            img = self.img_files[index]
            assert img is not None, 'File Not Found ' + img_path
            if self.n < 3000:  # cache into memory if image count < 3000
                self.imgs[index] = img

        # Load labels
        labels = []
        if os.path.isfile(self.path):
            x = self.bbs[index]
            if x is None:  # labels not preloaded
                with open(self.path, 'r') as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                    self.labels[index] = x  # save for next time
        labels = x
        labels = np.array(labels)
        labels_out = torch.zeros(1)
        labels_out = torch.from_numpy(labels)

        # Load ids
        id = []
        if os.path.isfile(self.path):
            x = self.ids[index]
            if x is None:  # labels not preloaded
                with open(self.path, 'r') as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                    self.labels[index] = x  # save for next time
        id = x
        id = np.array(id)
        id_out = torch.zeros(1)
        id_out = torch.from_numpy(id)
        '''
        print('img')
        print(img)
        print('label')
        print(labels_out)
        '''
        return torch.from_numpy(img), labels_out, id_out

    @staticmethod
    def collate_fn(batch):
        img, label, id = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), torch.cat(id, 0)
