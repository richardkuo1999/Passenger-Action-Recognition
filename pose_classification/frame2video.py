import cv2
import numpy as np
import glob
import shutil
import os

def last_chars(x):
    #print(x[-21:-16])
    #return(x[-21:-16])
    xext = x.split('.')[1]
    x = x.split('.')[0]
    x = "%04d"%int(x)
    #x = x + '.' + xext
    #print(x)
    return(x)

dirPath = "gt_clas_result/"

result = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
#results = sorted(result, key = last_chars)
results = sorted(result)

#print(results)
img_array = []
for filename in results:
    img = cv2.imread(dirPath + '/' + filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

#print(img_array)


out = cv2.VideoWriter('cy.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
