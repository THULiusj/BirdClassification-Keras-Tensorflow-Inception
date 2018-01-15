# Preprocess the data, split the images(in "images" folder) into training dataset and test dataset
# accoring to the tag in "train_test_split.txt" 
import pandas
import numpy as np
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from PIL import Image

IM_SIZE = (299, 299)

data = np.loadtxt("train_test_split.txt", delimiter=' ')
data = np.array(data, dtype=int)
imglist = np.genfromtxt('images.txt',delimiter=' ',dtype='str')

X_train = []
y_train = []
X_test=[]
y_test=[]
for imgid, flag in data:
   if flag == 0:
     X_train.append('images/'+ imglist[imgid-1, 1])
     y_train.append(imglist[imgid-1, 1].split("/",1)[0] )
   else:
     X_test.append('images/' + imglist[imgid-1, 1])
     y_test.append(imglist[imgid-1, 1].split("/",1)[0] )

# save the images to train dataset and test dataset
for imgpath in X_train:
    im = Image.open(imgpath)
    #im.thumbnail(IM_SIZE)
    savepath = 'train_images/'+ imgpath.split("/",1)[1].split("/",1)[0]
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    im.save(savepath + '/'+ imgpath.split("/",1)[1].split("/",1)[1])

for imgpath in X_test:
   im = Image.open(imgpath)
   #im.thumbnail(IM_SIZE)
   savepath = 'test_images/'+ imgpath.split("/",1)[1].split("/",1)[0]
   if not os.path.exists(savepath):
     os.makedirs(savepath)
   im.save(savepath + '/'+ imgpath.split("/",1)[1].split("/",1)[1])
