# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:38:29 2020

@author: Admin
"""

import os

lis = os.listdir('D:/action/action_detection/data/cam1-10-low')

with open('filenames.txt', 'a') as file:
    for i, fn in enumerate(lis):
        fn = fn.split('.')[0]
        file.write(fn+'\n')