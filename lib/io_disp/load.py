#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import dispmap
import matplotlib.pyplot as plt

db = dispmap.DisparityMap()

def disp2array(disp):
    w = disp.width()
    h = disp.height()
    res = np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            res[y,x] = disp.get_pixel(x,y)
    return res

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
db.writeColor('frompyColor.png',-1.0)
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



#print 'write to frompy.png'
#db.write('frompy.png')

