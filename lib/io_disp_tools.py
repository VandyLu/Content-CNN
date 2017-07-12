#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('./lib/io_disp')
import dispmap

def disp2array(disp,dtype=np.float32):
    w = disp.width()
    h = disp.height()
    res = np.zeros((h,w),dtype)
    for y in range(h):
        for x in range(w):
            res[y,x] = disp.get_pixel(x,y)
    return res

def make_disp(w,h):
    '''make an empty dispmap'''
    r = dispmap.DisparityMap()
    r.setSize(w,h)
    return r

def dispFromFile(fp):
    d = dispmap.DisparityMap()
    d.readPNG(fp)
    return d

def dispFromArray(img):
    d = make_disp(img.shape[1],img.shape[0])
    for i in range(d.height()):
        for j in range(d.width()):
            d.setData(img[i,j],j,i)

def copy_disp(disp):
    d = make_disp(disp.width(),disp.height())
    for i in range(d.height()):
        for j in range(d.width()):
            d.setData(disp.get_pixel(j,i),j,i)

if __name__=='__main__':
    db = dispmap.DisparityMap()
    
    path = '/data/stereo/data_scene_flow/training/disp_noc_0/'
    db.readPNG(path+'000000_10.png')
    
    print 'w:{},h{}\n'.format(db.width(),db.height())
    
    print 'test max disp:'
    img = disp2array(db)
    print 'python:',np.max(img)
    print 'c++:',db.maxDisp()
    print '\n\n'
    
    print 'test png display:'
    plt.imshow(img)
    plt.show()
    print 'done!\n\n'
    print 'test interpolateBackground'
    db.interpolateBackground()
    plt.imshow(disp2array(db))
    plt.show()
    print 'done!\n\n'
    
    print 'test writeColor:'
    #db.writeColor('frompyColor.png',-1.0)
    print 'done!\n\n'
    
    print 'test setWidth setData'
    img = np.random.rand(300,500)
    db.setSize(img.shape[1],img.shape[0])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            db.setData(img[i,j],j,i)
    plt.imshow(disp2array(db))
    plt.show()
    
    print 'after reassign'
    print 'w:{},h:{}'.format(db.width(),db.height())
    
    
