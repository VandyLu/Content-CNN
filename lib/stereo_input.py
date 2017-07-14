#!/usr/bin/python
# -*- coding: utf-8 -*-


import tensorflow as tf
import cv2
import numpy as np
import random
import sys
sys.path.append('./')
sys.path.append('./lib')
import cfg
import db



class StereoInput():

    def __init__(self,dbfile,batch_size=4):
        self.db_train = db.DB(mode='r')
        img0,img1,disp = self.db_train.read(dbfile['TRAIN'])
        self.img0,self.img1,self.disp = tf.train.shuffle_batch(    
            [img0,img1,disp],
            batch_size=batch_size,
            capacity=100,
            min_after_dequeue=20)

        self.db_test = db.DB(mode='r')
        img0,img1,disp = self.db_test.read(dbfile['TEST'])
        self.img0_test,self.img1_test,self.disp_test=tf.train.shuffle_batch( 
            [img0,img1,disp],
            batch_size=batch_size,
            capacity=100,
            min_after_dequeue=20)
    
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
        
