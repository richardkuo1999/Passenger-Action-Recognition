import os
import shutil
from random import sample


files_path1 = 'D:/action/pose_classification/data/labeled_winbus/cam1/cam1-GH010034-1_labeled_000'
files_path2 = 'D:/action/pose_classification/data/labeled_winbus/cam2/cam2-GH010218-1_labeled_000'
save_folder = 'C:/Users/user/Desktop/123.txt'
files_all = os.listdir(files_path1)
files_list = []
f= open(save_folder,'w')
for files in files_all:
    if((files.split('.')[1] == 'jpg')):
        files_list.append(files.split('.')[0])
        f.write('data/labeled_winbus/cam1/cam1-GH010034-1_labeled_000/'+files+'\n')
files_all = os.listdir(files_path2)
for files in files_all:
    if((files.split('.')[1] == 'jpg')):
        files_list.append(files.split('.')[0])
        f.write('data/labeled_winbus/cam2/cam2-GH010218-1_labeled_000/'+files+'\n')

f.close()


            





                