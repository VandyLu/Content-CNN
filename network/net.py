#!/usr/bin/python
# -*- coding: utf-8 -*-

# Efficient Deep Learning for Stereo Matching(Luo)
# 

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time
import sys
sys.path.append('./')
sys.path.append('./lib')
import cfg
import db
import stereo_input

batch_norm = tf.contrib.layers.batch_norm
convolution2d = tf.contrib.layers.convolution2d

def variable_summaries(var,name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name,var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name,mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev/'+name,stddev)

def getWindow(img0,img1,disp,center,size):
    ''' img0: H*W*3
        disp: H*W
        size: int 
        center: tensor n*2
    '''
    dispmax = cfg.param.dispmax
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
    dispmax = cfg.param.dispmax
    size = cfg.param.window_size
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

    
def roiLayer(img0,img1,disp,n):
    ''' pick n blocks from img0 and img1 where ground truth is available
        img0,img1: b* H*W*C
        disp: b*H*W
        return: ((b*n) *...)
    '''
    batch_size = tf.shape(img0)[0]
    dispmax = cfg.param.dispmax
    size = cfg.param.window_size

    displist = tf.unstack(disp)
    img0list = tf.unstack(img0)
    img1list = tf.unstack(img1)
    
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

   
def test_roiLayer():
    data_path = {
        'train':'./data/train.tfrecords',
        'test':'./data/val.tfrecords'} 

    si = stereo_input.StereoInput(data_path,batch_size=4)
    img0_tr,img1_tr,disp_tr = si.train_batch()
    img0_te,img1_te,disp_te = si.test_batch()

    img0,img1,disp = roiLayer(img0_tr,img1_tr,disp_tr,16)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        # warm up
        i0,i1,d = sess.run([img0_tr,img1_tr,disp_tr])
        cv2.imshow('i0',i0[0].astype(np.uint8))
        cv2.imshow('i1',i1[0].astype(np.uint8))
        cv2.imshow('d',d[0].astype(np.uint8))
        cv2.waitKey()
        for i in range(3):
            i0,i1,d = sess.run([img0,img1,disp])
            print 'img0 shape:',i0.shape
            print 'img1 shape:',i1.shape
            print 'disp shape:',d.shape

            cv2.imshow('i0',i0[0].astype(np.uint8))
            cv2.imshow('i1',i1[0].astype(np.uint8))
            print d
            cv2.waitKey()

        t1 = time.time()
        n = 50
        for i in range(n):
            i0,i1,d = sess.run([img0,img1,disp])
        t2 = time.time()
        print '{}s/batch'.format((t2-t1)/n)
        coord.request_stop()
        coord.join(threads)


initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)

def conv2d(x, kernel_shape, strides=[1,1,1,1], relu=True,padding='SAME',reuse=False,scope=None):
    kernel = kernel_shape[0]
    input_nums = kernel_shape[2]
    output_nums = kernel_shape[3]
    return convolution2d(x,output_nums,kernel,normalizer_fn=batch_norm,activation_fn=tf.nn.relu,padding=padding,
            weights_initializer=initializer,biases_initializer=initializer,reuse=reuse,scope=scope)
    

#def conv2d(x, kernel_shape, strides=[1,1,1,1], relu=True, padding='SAME'):
#    W = tf.get_variable("weights", kernel_shape, initializer=initializer)
#    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
#    b = tf.get_variable("biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))
#    with tf.name_scope("conv"):
#        # attention !!!
#        # disparity from 0-227  strides[3] must be 1
#        x = tf.nn.conv2d(x, W, strides=strides, padding=padding)
#        x = tf.nn.bias_add(x, b)
#        if kernel_shape[2] == 3:
#            x_min = tf.reduce_min(W)
#            x_max = tf.reduce_max(W)
#            kernel_0_to_1 = (W - x_min) / (x_max - x_min)
#            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
#            tf.summary.image('filters', kernel_transposed, max_outputs=3)
#        if relu:
#            x = tf.nn.relu(x)
#            #x = tf.maximum(LEAKY_ALPHA * x, x)
#    return x

def resblock(x0,kernel_shape):
    W1 = tf.get_variable("weights1",kernel_shape,initializer=initializer)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,W1)
    b1 = tf.get_variable("biases1",kernel_shape[3],initializer=tf.constant_initializer(0.0))

    W2 = tf.get_variable("weights2",kernel_shape,initializer=initializer)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,W2)
    b2 = tf.get_variable("biases2",kernel_shape[3],initializer=tf.constant_initializer(0.0))

    with tf.name_scope("resblock"):
       x = tf.nn.conv2d(x0,W1,strides=[1,1,1,1],padding='SAME')
       x = tf.nn.bias_add(x,b1)
       x = tf.nn.relu(x)
       x = tf.nn.conv2d(x,W2,strides=[1,1,1,1],padding='SAME')
       x = tf.nn.bias_add(x,b2)
       x = x + x0
       x = tf.nn.relu(x)
    return x

#def shift(x,dispmax):
#    ''' x is a tensor like(n,32,32+228,1) 
#        make a (1,32,32,1) window
#        slide and get (n,228,32,32)
#    '''
#    batch_size = tf.shape(x)[0]
#    H = tf.shape(x)[1]
#    size = [batch_size,H,H,1]
#    shiftList = []
#    for i in range(dispmax,0,-1):
#        shiftList.append(tf.slice(x,[0,0,i-1,0],size))
#    return tf.stack(shiftList,axis=1)

def shift(x,dispmax):
    ''' input (n,32,32+dispmax-1,1)
        return (n*dispmax,32,32,1)'''
    #H = tf.shape(x)[1]
    #C = tf.shape(x)[3]
    H = cfg.param.window_size
    imglist = tf.unstack(x)
    xlist = []
    size = [H,H,-1]
    for img in imglist:
        for i in range(dispmax,0,-1):
            xlist.append(tf.slice(img,[0,i-1,0],size))
    return tf.stack(xlist)

def shift_loss(flata,flatb,disp,dispmax,p):
    ''' flata: (n,64)
        flatb: (n*228,64)
        disp: (n,)
        p: 3 pixel error distribution [0.5,0.2,0.05,0]
        return loss
    '''
    alist = tf.unstack(flata)
    blist = tf.split(flatb,len(alist),0)
    dlist = tf.unstack(disp)

    all_probs = []
    probs = []
    argmins = []
    for a,b in zip(*(alist,blist)):
        prob = tf.nn.softmax(tf.matmul(b,tf.expand_dims(a,-1)))
        argmin = tf.cast(tf.argmin(tf.reshape(prob,[-1])),tf.int32)
        all_probs.append(prob)
        probs.append(prob[argmin])
        argmins.append(argmin)

    #for a,b in zip(*(alist,blist)):
    #    prob = tf.nn.softmax(tf.matmul(b,tf.expand_dims(a,-1)))
    #    all_probs.append(prob)
    #all_probs = tf.stack(all_probs)
    #argmins = tf.argmax(all_probs)
    #probs = tf.maximum(probs)
    #p_i = p[tf.minimum(tf.abs(argmins-disp),3)]

    losses = []
    for y,gt,prob in zip(*(argmins,dlist,probs)):
        p_i = tf.gather(p,tf.minimum(tf.abs(y-gt),3))
        losses.append(-(p_i*tf.log(prob)+(1.0-p_i)*tf.log(1.0-prob)))

    loss = tf.reduce_mean(tf.stack(losses))
    all_probs = tf.stack(all_probs)
    argmins = tf.stack(argmins)
    return loss,all_probs,argmins

def traintest_pipeline(batch_size):
    path = cfg.param.dataset
    si = stereo_input.StereoInput(path,batch_size=batch_size)
    train_batch = si.train_batch()
    test_batch = si.test_batch()
    
    return (train_batch,test_batch)

# not implemented yet
def val_pipeline():
    path = cfg.param.dataset
    si = stereo_input.StereoInput(path,batch_size=1)
    train_batch = si.train_batch()
    test_batch = si.test_batch()
    return test_batch


class Luo():
    def __init__(self,mode,ckpt_path=None):
        '''
            mode: 'TRAIN' 'TEST'
        '''
        self.mode = mode
        self.crop_shape = cfg.param.crop_shape
        self.window_size = cfg.param.window_size
        self.dispmax = cfg.param.dispmax
        self.lr = cfg.param.learning_rate
        self.beta1 = cfg.param.beta1
        self.beta2 = cfg.param.beta2
        self.img2batch = cfg.param.img2batch

        self.build_model()
    def build_model(self,scope='Luo'):
        self.global_step = tf.Variable(0,trainable=False)
        self.p = tf.constant([0.5,0.2,0.05,0.0])
        
        t0 = time.time()
        with tf.name_scope('data_roi'):
            if self.mode=='TRAIN':
                train_batch,test_batch = traintest_pipeline(batch_size=2) # 4*32=128
                train_batch = roiLayer(train_batch[0],train_batch[1],train_batch[2],self.img2batch)
                test_batch = roiLayer(test_batch[0],test_batch[1],test_batch[2],32)
                self.inputs = train_batch
            elif self.mode=='VAL':
                self.x0 = tf.placeholder(tf.float32,[None,self.window_size,self.window_size,3])
                self.x1 = tf.placeholder(tf.float32,[None,self.window_size,self.window_size+self.dispmax-1,3])
                self.y = tf.placeholder(tf.float32,[None])
                self.inputs = (self.x0,self.x1,self.y)
        t1 = time.time()
        print 'input:{}s'.format(t1-t0)

        img0_batch,img1_batch,disp_batch = self.inputs

        if cfg.param.kernel == 5:
            # (n,21,21,3)
            # (n,21,21+227,3)
            with tf.variable_scope('conv1') as scope:
                conv1a = conv2d(img0_batch,[5,5,3,64],strides=[1,1,1,1],padding='VALID')
                scope.reuse_variables()
                conv1b = conv2d(img1_batch,[5,5,3,64],strides=[1,1,1,1],padding='VALID')
            # (n,17,17+227,64)
            with tf.variable_scope('conv2') as scope:
                conv2a = conv2d(conv1a,[5,5,64,64],strides=[1,1,1,1],padding='VALID')
                scope.reuse_variables()
                conv2b = conv2d(conv1b,[5,5,64,64],strides=[1,1,1,1],padding='VALID')
            # (n,13,13+227,64)
            with tf.variable_scope('conv3') as scope:
                conv3a = conv2d(conv2a,[5,5,64,64],strides=[1,1,1,1],padding='VALID')
                scope.reuse_variables()
                conv3b = conv2d(conv2b,[5,5,64,64],strides=[1,1,1,1],padding='VALID')
            # (n,9,9+227,64)
            with tf.variable_scope('conv4') as scope:
                conv4a = conv2d(conv3a,[5,5,64,64],strides=[1,1,1,1],padding='VALID')
                scope.reuse_variables()
                conv4b = conv2d(conv3b,[5,5,64,64],strides=[1,1,1,1],padding='VALID')
            # (n,5,5+227,64)
            with tf.variable_scope('conv5') as scope:
                conv5a = conv2d(conv4a,[5,5,64,64],strides=[1,1,1,1],padding='VALID')
                scope.reuse_variables()
                conv5b = conv2d(conv4b,[5,5,64,64],strides=[1,1,1,1],padding='VALID')
            # (n,1,228,64)
        elif cfg.param.kernel == 0:
            # (n,9,9,3)
            # (n,9,9+227,3)
            with tf.variable_scope('conv1') as scope:
                conv1a = conv2d(img0_batch,[3,3,3,64],strides=[1,1,1,1],padding='VALID')
                scope.reuse_variables()
                conv1b = conv2d(img1_batch,[3,3,3,64],strides=[1,1,1,1],padding='VALID')
            # (n,7,7+227,64)
            with tf.variable_scope('conv2') as scope:
                conv2a = conv2d(conv1a,[3,3,64,64],strides=[1,1,1,1],padding='VALID')
                scope.reuse_variables()
                conv2b = conv2d(conv1b,[3,3,64,64],strides=[1,1,1,1],padding='VALID')
            # (n,5,5+227,64)
            with tf.variable_scope('conv3') as scope:
                conv3a = conv2d(conv2a,[3,3,64,64],strides=[1,1,1,1],padding='VALID')
                scope.reuse_variables()
                conv3b = conv2d(conv2b,[3,3,64,64],strides=[1,1,1,1],padding='VALID')
            # (n,3,3+227,64)
            with tf.variable_scope('conv4') as scope:
                conv5a = conv2d(conv3a,[3,3,64,64],strides=[1,1,1,1],padding='VALID')
                scope.reuse_variables()
                conv5b = conv2d(conv3b,[3,3,64,64],strides=[1,1,1,1],padding='VALID')
            # (n,1,1+227,64)
        elif cfg.param.kernel ==3:
            with tf.variable_scope('conv1') as scope:
                conv1a = conv2d(img0_batch,[3,3,3,64],strides=[1,1,1,1],padding='VALID',scope=scope)
                conv1b = conv2d(img1_batch,[3,3,3,64],strides=[1,1,1,1],padding='VALID',reuse=True,scope=scope)
            # (n,7,7+227,64)
            with tf.variable_scope('conv2') as scope:
                conv2a = conv2d(conv1a,[3,3,64,64],strides=[1,1,1,1],padding='VALID',scope=scope)
                conv2b = conv2d(conv1b,[3,3,64,64],strides=[1,1,1,1],padding='VALID',reuse=True,scope=scope)
            # (n,5,5+227,64)
            with tf.variable_scope('conv3') as scope:
                conv3a = conv2d(conv2a,[3,3,64,64],strides=[1,1,1,1],padding='VALID',scope=scope)
                conv3b = conv2d(conv2b,[3,3,64,64],strides=[1,1,1,1],padding='VALID',reuse=True,scope=scope)
            # (n,3,3+227,64)
            with tf.variable_scope('conv4') as scope:
                conv5a = conv2d(conv3a,[3,3,64,64],strides=[1,1,1,1],padding='VALID',scope=scope)
                conv5b = conv2d(conv3b,[3,3,64,64],strides=[1,1,1,1],padding='VALID',reuse=True,scope=scope)
            # (n,1,1+227,64)

        with tf.name_scope('reshapes'):
            va = tf.reshape(conv5a,[-1,64,1])
            vb = tf.reshape(conv5b,[-1,228,64])
            vb = tf.reverse(vb,[1]) # to make the image disp==0 located at the [0]
        print 'va:',va
        print 'vb:',vb

        with tf.name_scope('dot'):
            self.all_probs = tf.clip_by_value(tf.nn.softmax(tf.reshape(tf.matmul(vb,va),(-1,228))),1e-12,1.0)
            tf.summary.histogram('disparity_probs',self.all_probs)

        batch_size = tf.shape(disp_batch)[0]
        with tf.name_scope('gather'):
            self.gt = tf.cast(disp_batch,tf.int32) #disp==0 vb[227]
            all_p = tf.reshape(self.all_probs,[-1])
            gather_index1 = self.dispmax*tf.range(batch_size)+self.gt
            gather_index2 = self.dispmax*tf.range(batch_size)+tf.maximum(self.gt+1,self.dispmax-1)
            gather_index3 = self.dispmax*tf.range(batch_size)+tf.maximum(self.gt+2,self.dispmax-1)
            gather_index2_ = self.dispmax*tf.range(batch_size)+tf.minimum(self.gt-1,0)
            gather_index3_ = self.dispmax*tf.range(batch_size)+tf.minimum(self.gt-2,0)

            self.probs = tf.gather(all_p,gather_index1)
            self.probs2 = tf.gather(all_p,gather_index2)
            self.probs2_ = tf.gather(all_p,gather_index2_)
            self.probs3 = tf.gather(all_p,gather_index3)
            self.probs3_ = tf.gather(all_p,gather_index3_)

        with tf.name_scope('loss'):
            self.losses = -(0.5*tf.log(self.probs)+0.2*tf.log(self.probs2)+0.2*tf.log(self.probs2_)+
                0.05*tf.log(self.probs3) + 0.05*tf.log(self.probs3_))
            self.loss = tf.reduce_mean(self.losses)
            tf.summary.scalar('cross-entropy',self.loss)
        
        with tf.name_scope('accuracy'):
            self.predictions = tf.cast(tf.argmax(self.all_probs,1),tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.less(tf.abs(self.predictions-self.gt),3),tf.float32))
            tf.summary.scalar('acc',self.accuracy)
#lr = tf.train.exponential_decay(self.lr,global_step=self.global_step,decay_steps=100,decay_rate=0.95)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                            beta1=self.beta1,
                                            beta2=self.beta2)
        self.train_op = self.optimizer.minimize(self.loss,global_step=self.global_step)
        self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    
    net = Luo(mode='TRAIN')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        _,loss,accuracy = sess.run([net.train_op,net.loss,net.accuracy])
        print loss
        print accuracy

        coord.request_stop()
        coord.join(threads)


