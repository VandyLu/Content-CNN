#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('./')
import cfg

def processLeft(img):
    crop = cfg.param.crop_shape
    img = img[0:crop[0],0:crop[1]]
    img = (img-np.mean(img))/np.std(img)
    return img

def processRight(img):
    return processLeft(img)

def processDisp(img):
    ''' disp \in [0,70]
    '''
    crop = cfg.param.crop_shape
    img = img[0:crop[0],0:crop[1]]
    return img
    


if __name__=='__main__':
    path = '/home/lfb/Documents/DataSet/stereo/data_scene_flow/training/disp_noc_0/'
    if 1:
        m = -1
        a = 10e5
        for f in os.listdir(path):
            img = cv2.imread(path+f,-1)
            print np.max(processDisp(img))
            d = img[img!=0]
            mm = np.max(img)
            bb = np.min(img)
            if mm>m:
                m = mm 
            if bb<a:
                a = bb
        print 'max:',m
        print 'min:',a
        d.sort()
        plt.plot(np.arange(d.shape[0]),d,'r*')
        plt.show()
    img = cv2.imread(path+'000000_10.png',-1)
    print np.max(processDisp(img))
    
