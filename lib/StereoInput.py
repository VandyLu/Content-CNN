#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import random
import os
import sys
sys.path.append('./')
sys.path.append('./lib')
import cfg
import db

def getWindow(img0,img1,disp,center,size):
    ''' img0: H*W*3
        disp: H*W
        size: int 
        center: tensor n*2
    '''
    dispmax = cfg.dispmax
    cs = tf.unstack(center)
    img0_batch = []
    img1_batch = []
    disp_batch = []
    for c in cs:
        slice_start = tf.cast(c - size/2,tf.int32)
        img0_batch.append(tf.slice(img0,[slice_start[0],slice_start[1],0],(size,size,-1)))
        slice_start = tf.cast(c - [size/2,size/2+dispmax-1],tf.int32)
        img1_batch.append(tf.slice(img1,[slice_start[0],slice_start[1],0],(size,size+dispmax-1,-1)))
        disp_batch.append(disp[c[0],c[1]])
    return img0_batch,img1_batch,disp_batch

def roiCrop(img0,img1,disp,n):
    dispmax = cfg.dispmax
    size = cfg.window_size
    pad_size = dispmax + size
    img0 = tf.pad(img0,[[pad_size,pad_size],[pad_size,pad_size],[0,0]],'CONSTANT')
    img1 = tf.pad(img1,[[pad_size,pad_size],[pad_size,pad_size],[0,0]],'CONSTANT')
    disp = tf.pad(disp,[[pad_size,pad_size],[pad_size,pad_size]],'CONSTANT')

    valid = tf.where(disp>0)
    #for i in range(n):
    center_index = tf.random_uniform((n,),minval=0,maxval=tf.shape(valid)[0],dtype=tf.int32)
    center = tf.cast(tf.gather(valid,center_index),tf.int32)
    
    img0_batch,img1_batch,disp_batch=getWindow(img0,img1,disp,center,size)
    return (img0_batch,img1_batch,disp_batch)

    
def extract_patches(img0,img1,disp,n):
    ''' pick n blocks from img0 and img1 where ground truth is available
        img0,img1: b* H*W*C (uint8)
        disp: b*H*W
        return: ((b*n)*H*W*C,(b*n)*H*W*C,(b*n))
    '''
    batch_size = tf.shape(img0)[0]
    dispmax = cfg.dispmax
    size = cfg.window_size
    img_num = cfg.img_num 
    displist = tf.unstack(disp,img_num)
    img0list = tf.unstack(img0,img_num)
    img1list = tf.unstack(img1,img_num)
    
    img0_batch = []
    img1_batch = []
    disp_batch = []

    for i0,i1,d in zip(*(img0list,img1list,displist)):
        img0_batch_,img1_batch_,disp_batch_ = roiCrop(i0,i1,d,n)
        img0_batch.extend(img0_batch_)
        img1_batch.extend(img1_batch_)
        disp_batch.extend(disp_batch_)

    img0_batch = tf.stack(img0_batch)
    img1_batch = tf.stack(img1_batch)
    disp_batch = tf.stack(disp_batch)
    return (img0_batch,img1_batch,disp_batch) 

def train_test_pipeline(batch_size):
    si = StereoInput(cfg.dataset,cfg.img_num)
    return si.train_batch(),si.test_batch()

def val_pipeline(batch_size):
    img_name = os.listdir(cfg.val_img0)
    img_name = np.random.choice(img_name,batch_size)
    batch = [ (cv2.imread(cfg.val_img0+name,-1)[0:300,0:1000],cv2.imread(cfg.val_img1+name,-1)[0:300,0:1000]) for name in img_name ]
    img0,img1 = zip(*batch)
    return np.stack(img0),np.stack(img1)


class StereoInput():
    def __init__(self,dbfile,img_num=4):
        self.db_train = db.DB(mode='r')

        img0,img1,disp = self.db_train.read(dbfile['TRAIN'])
        img0,img1,disp = tf.train.shuffle_batch(    
            [img0,img1,disp],
            batch_size=img_num,
            capacity=100,
            min_after_dequeue=20)
        #with tf.name_scope('train_pipeline'):
        #    self.img0,self.img1,self.disp = extract_patches(img0,img1,disp,batch_size/img_num)
        self.img0,self.img1,self.disp = img0,img1,disp

        self.db_test = db.DB(mode='r')
        img0,img1,disp = self.db_test.read(dbfile['TEST'])
        img0,img1,disp=tf.train.shuffle_batch( 
            [img0,img1,disp],
            batch_size=img_num,
            capacity=100,
            min_after_dequeue=20)
        #with tf.name_scope('test_pipline'):
        #    self.img0_test,self.img1_test,self.disp_test = extract_patches(img0,img1,disp,batch_size/img_num)
        self.img0_test,self.img1_test,self.disp_test = img0,img1,disp
    
    def train_batch(self):
        return (self.img0,self.img1,self.disp)
    def test_batch(self):
        return (self.img0_test,self.img1_test,self.disp_test)
    

    
if __name__=='__main__':
    data_path = {
        'TRAIN':'./data/train.tfrecords',
        'TEST':'./data/val.tfrecords'} 
        
    si = StereoInput(data_path,batch_size=4)
    img0_tr,img1_tr,disp_tr = si.train_batch()
    img0_te,img1_te,disp_te = si.test_batch()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        img0,img1,disp = sess.run([img0_tr,img1_tr,disp_tr])
        print type(disp)
        coord.request_stop()
        coord.join(threads)

    print img0[0]
    print np.sum(np.abs(img0[0]-img1[0]))
    cv2.imshow('img0',img0[0].astype(np.uint8))
    cv2.imshow('img1',img1[0].astype(np.uint8))
    cv2.waitKey()
    d = disp[0]
    print d.shape
    print np.max(d)
    print np.min(d)
    i0 = img0[0]
    print i0.shape
    print np.max(i0)
    print np.min(i0)
        
