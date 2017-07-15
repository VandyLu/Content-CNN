#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools

sys.path.append('./')
sys.path.append('./lib')
sys.path.append('./lib/io_disp')
import cfg
import preprocess as pre
#from io_disp_tools import *


class Pair():
    def __init__(self,img0,img1,disp,preprocess = True):
        self.img0 = img0.astype(np.uint8)
        self.img1 = img1.astype(np.uint8)
        self.disp = disp.astype(np.float32)
        if preprocess:
            self.img0 = pre.processLeft(self.img0)
            self.img1 = pre.processRight(self.img1)
            self.disp = pre.processDisp(self.disp)
        
        self.size = self.disp.shape
        print self.size

    def tobytes(self):
        return (self.img0.tobytes(),self.img1.tobytes(),self.disp.tobytes())

class DB():
 
    def __init__(self,mode='r',**param):
        self.mode = mode
        self.param = param
        
    def read(self,dbfile):
        assert os.path.isfile(dbfile),'Read {} failed'.format(dbfile)
        producer = tf.train.string_input_producer([dbfile])
        self.reader = tf.TFRecordReader()
        _,eg = self.reader.read(producer)
        features = tf.parse_single_example(eg,features={
                'img0': tf.FixedLenFeature([],tf.string),
                'img1': tf.FixedLenFeature([],tf.string),
                'disp': tf.FixedLenFeature([],tf.string),
                }
            )
        shape = cfg.param.crop_shape
        img0 = tf.decode_raw(features['img0'],tf.uint8)
        img0 = tf.reshape(img0,(shape[0],shape[1],3))
        img1 = tf.decode_raw(features['img1'],tf.uint8)
        img1 = tf.reshape(img1,(shape[0],shape[1],3))
        disp = tf.decode_raw(features['disp'],tf.float32)
        disp = tf.reshape(disp,(shape[0],shape[1]))

        self.read_img0 = img0
        self.read_img1 = img1
        self.read_disp = tf.cast(disp,tf.float32) 
        return (self.read_img0,self.read_img1,self.read_disp) # tensors

        
    def write(self,dbfile):
        '''
            dbfile: {'train':p,'test':p} eg. ./data/a.tfrecords
        '''
        self.write_init_(shuffle=False) # ln -s path data
        self.check_db_exist_(dbfile['train'])
        self.check_db_exist_(dbfile['test'])
        train_writer = tf.python_io.TFRecordWriter(dbfile['train'])
        val_writer = tf.python_io.TFRecordWriter(dbfile['test'])
        
        for name in self.img_tr:
            pair = self.getPair_(name,preprocess=True)
            img0_raw,img1_raw,disp_raw = pair.tobytes()
            example = tf.train.Example(features = tf.train.Features(
                feature={
                    'img0': tf.train.Feature( bytes_list=tf.train.BytesList(value=[img0_raw]) ),
                    'img1': tf.train.Feature( bytes_list=tf.train.BytesList(value=[img1_raw]) ),
                    'disp': tf.train.Feature( bytes_list=tf.train.BytesList(value=[disp_raw]) )
                    }
                ))
            train_writer.write(example.SerializeToString())   

        for name in self.img_te:
            pair = self.getPair_(name,preprocess=True)
            img0_raw,img1_raw,disp_raw = pair.tobytes()
            example = tf.train.Example(features = tf.train.Features(
                feature={
                    'img0': tf.train.Feature( bytes_list=tf.train.BytesList(value=[img0_raw]) ),
                    'img1': tf.train.Feature( bytes_list=tf.train.BytesList(value=[img1_raw]) ),
                    'disp': tf.train.Feature( bytes_list=tf.train.BytesList(value=[disp_raw]) )
                    }
                ))
            val_writer.write(example.SerializeToString())
        
        train_writer.close()
        val_writer.close()
        print 'write DB done'
        

    def check_db_exist_(self,dbfile):
        if os.path.exists(dbfile):
            check = raw_input('do you want to delete {}'.format(dbfile))
            if not check == 'y':
                print 'Errorin create_db_'
                return 
            else:
                os.remove(dbfile)

    
    def write_init_(self,path=None,shuffle=False,param=('noc')):
        ''' path is the dir like this: path/training/image_2'''
        if path==None:
            path = os.path.join(os.getcwd(),'data/data_scene_flow')
        assert os.path.isdir(path),"Dataset path not exist: "+path

        self.dir = path
        self.left_dir = os.path.join(self.dir,'training/image_2') 
        self.right_dir = os.path.join(self.dir,'training/image_3')
        if 'noc' in param:
            self.disp_left_dir = os.path.join(self.dir,'training/disp_noc_0')
            self.disp_right_dir = os.path.join(self.dir,'training/disp_noc_1')
        else:
            self.disp_left_dir = os.path.join(self.dir,'training/disp_occ_0')
            self.disp_right_dir = os.path.join(self.dir,'training/disp_occ_1')

        self.img_set = [os.path.splitext(s)[0] for s in os.listdir(self.disp_left_dir)]
        
        # cfg
        self.n_tr = cfg.param.n_tr
        self.n_te = cfg.param.n_te
        if self.n_tr+self.n_te > 200:
            print 'Warning: n_tr+n_te >200'
        if shuffle:
            np.random.shuffle(self.img_set)
        self.img_tr = self.img_set[0:self.n_tr]
        self.img_te = self.img_set[self.n_tr:(self.n_tr+self.n_te)]
        
        print 'Dataset:{},num:{}'.format(self.dir,len(self.img_set))
        print '{} for training, {} for testing'.format(self.n_tr,self.n_te)

    def getPair_(self,name,suffix='.png',preprocess=False):
        img0 = cv2.imread(os.path.join(self.left_dir,name+suffix),-1)
        img1 = cv2.imread(os.path.join(self.right_dir,name+suffix),-1)

        gt_disp0 = disp2array(dispFromFile(os.path.join(self.disp_left_dir,name+suffix)))
        #gt_disp0 = cv2.imread(os.path.join(self.disp_left_dir,name+suffix),-1)
        #gt_disp1 = cv2.imread(os.path.join(self.disp_right_dir,name+suffix),-1)
        return Pair(img0,img1,gt_disp0,preprocess=preprocess)

    def getPair_normal(self,i):
        return self.getPair_(self.img_set[i],preprocess=True)

if __name__=='__main__':
    db = DB(mode='w') 

    data_path = {
        'train':'./data/train.tfrecords',
        'test':'./data/val.tfrecords'}
    db.write(data_path)
    print 'db'+'*'*30

    img0,img1,disp = db.read('./data/val.tfrecords')
    s = tf.shape(disp)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        i0,i1,d = sess.run([img0,img1,disp])
        print i0.shape
        print i1.shape
        print d.shape
        print d
        #sess.run([s])
        
    print 'test OK'

    #cv2.imshow('left',left_)
    #cv2.imshow('right',right_)
    #cv2.imshow('disp',disp_)
    #cv2.waitKey()

