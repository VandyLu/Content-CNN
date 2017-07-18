#!/usr/bin/python
# -*- coding: utf-8 -*-

# Efficient Deep Learning for Stereo Matching(Luo)

import tensorflow as tf
import numpy as np
import sys
sys.path.append('./')
sys.path.append('./lib')
import cfg
from StereoInput import extract_patches 
from StereoInput import train_test_pipeline
from StereoInput import val_pipeline

shift_corr_module = tf.load_op_library('./user_ops/shift_corr.so')
shift_corr = shift_corr_module.shift_corr 

batch_norm = tf.contrib.layers.batch_norm
convolution2d = tf.contrib.layers.convolution2d
initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)


class Content_CNN():
    def __init__(self):
        self.window_size = cfg.window_size # 9
        self.dispmax = cfg.dispmax
        self.beta1 = cfg.beta1
        self.beta2 = cfg.beta2
        self.train_batch,self.test_batch = train_test_pipeline(batch_size=cfg.batch_size)

        self.build_model()

    def build_model(self):
        self.mode = tf.placeholder_with_default(input=True,shape=(),name='train_or_not')
        self.global_step = tf.Variable(0,trainable=False)
	self.lr = tf.train.exponential_decay(cfg.learning_rate, self.global_step, cfg.decay_steps, cfg.decay_rate) 
        self.p = tf.constant([0.5,0.2,0.05,0.0])

        with tf.name_scope('inferece_inputs'): # inference branch
            self.val_img0 = tf.placeholder_with_default(input=tf.zeros([0,0,0,3],dtype=tf.uint8),
                                                        shape=[None,None,None,3])
            self.val_img1 = tf.placeholder_with_default(input=tf.zeros([0,0,0,3],dtype=tf.uint8),
                                                        shape=[None,None,None,3])
            self.val_disp = tf.placeholder_with_default(input=tf.zeros([1,1,1],dtype=tf.float32),shape=[None,None,None])
            pad_size = self.window_size / 2
            val_img0 = tf.pad(self.val_img0,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
            val_img1 = tf.pad(self.val_img1,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
            val_disp = tf.pad(self.val_disp,[[0,0],[pad_size,pad_size],[pad_size,pad_size]])

        with tf.name_scope('ROI'): # train branch
            self.img0,self.img1,self.disp = self.train_batch # origin images
            self.x0,self.x1,self.y = extract_patches(self.img0,self.img1,self.disp,cfg.batch_size/cfg.img_num)
            # if you want to check input images
            #tf.summary.image('input_x0',self.x0,10)
            #tf.summary.image('input_x1',self.x1,10)
            #tf.summary.histogram('input_disp',self.d)

        img0,img1,disp = tf.cond(self.mode,
                        lambda:(self.x0,self.x1,self.y),
                        lambda:(val_img0,val_img1,val_disp))

        img0 = tf.cast(img0,tf.float32)
        img1 = tf.cast(img1,tf.float32)

        with tf.variable_scope('conv1') as scope:
            conv1a = convolution2d(img0,64,3,normalizer_fn=batch_norm,activation_fn=tf.nn.relu,padding='VALID',weights_initializer=initializer,scope=scope)
            conv1b = convolution2d(img1,64,3,normalizer_fn=batch_norm,activation_fn=tf.nn.relu,padding='VALID',weights_initializer=initializer,reuse=True,scope=scope)

        with tf.variable_scope('conv2') as scope:
            conv2a = convolution2d(conv1a,128,3,normalizer_fn=batch_norm,activation_fn=tf.nn.relu,padding='VALID',weights_initializer=initializer,scope=scope)
            conv2b = convolution2d(conv1b,128,3,normalizer_fn=batch_norm,activation_fn=tf.nn.relu,padding='VALID',weights_initializer=initializer,reuse=True,scope=scope)
            
        with tf.variable_scope('conv3') as scope:
            conv3a = convolution2d(conv2a,64,3,normalizer_fn=batch_norm,activation_fn=tf.nn.relu,padding='VALID',weights_initializer=initializer,scope=scope)
            conv3b = convolution2d(conv2b,64,3,normalizer_fn=batch_norm,activation_fn=tf.nn.relu,padding='VALID',weights_initializer=initializer,reuse=True,scope=scope)

        with tf.variable_scope('conv4') as scope:
            conv4a = convolution2d(conv3a,64,3,normalizer_fn=batch_norm,activation_fn=tf.nn.relu,padding='VALID',weights_initializer=initializer,scope=scope)
            conv4b = convolution2d(conv3b,64,3,normalizer_fn=batch_norm,activation_fn=tf.nn.relu,padding='VALID',weights_initializer=initializer,reuse=True,scope=scope)
        # (n,1,1+227,64)

        # CPU implements, only for inference efficiency, no back propagate
        with tf.name_scope('shift_corr'):
            self.val_D_probs = tf.nn.softmax(shift_corr(conv4a,conv4b,self.dispmax))
            self.val_predictions = tf.argmax(self.val_D_probs,axis=-1)
            print 'val_D_probs:',self.val_D_probs
            print 'val_predictions:',self.val_predictions
                    
        with tf.name_scope('train'):
            with tf.name_scope('reshape'):
                va = tf.reshape(conv4a,[-1,64,1])
                vb = tf.reshape(conv4b,[-1,228,64])
                vb = tf.reverse(vb,[1]) # to make the image disp==0 located at the [0]
            print 'single_va:',va
            print 'single_vb:',vb
    
            with tf.name_scope('dot'):
                self.D_probs = tf.nn.softmax(tf.reshape(tf.matmul(vb,va),(-1,228)))
                self.D_probs = tf.clip_by_value(self.D_probs,1e-12,1.0) # avoid log(0)
            
            batch_size = tf.shape(self.x0)[0]
            with tf.name_scope('gather'):
                self.gt = tf.cast(disp,tf.int32)
                flat_D_probs = tf.reshape(self.D_probs,[-1])
    
                gather_index1 = self.dispmax*tf.range(batch_size)+self.gt
                gather_index2 = self.dispmax*tf.range(batch_size)+tf.minimum(self.gt+1,self.dispmax-1)
                gather_index3 = self.dispmax*tf.range(batch_size)+tf.minimum(self.gt+2,self.dispmax-1)
                gather_index2_ = self.dispmax*tf.range(batch_size)+tf.maximum(self.gt-1,0)
                gather_index3_ = self.dispmax*tf.range(batch_size)+tf.maximum(self.gt-2,0)
    
                self.probs = tf.gather(flat_D_probs,gather_index1) # prob of gt
                probs2 = tf.gather(flat_D_probs,gather_index2)
                probs2_ = tf.gather(flat_D_probs,gather_index2_)
                probs3 = tf.gather(flat_D_probs,gather_index3)
                probs3_ = tf.gather(flat_D_probs,gather_index3_)
    
            with tf.name_scope('loss'):
                self.losses = -(0.5*tf.log(self.probs)+0.2*tf.log(probs2)+0.2*tf.log(probs2_)+
                    0.05*tf.log(probs3) + 0.05*tf.log(probs3_))
                self.loss = tf.reduce_mean(self.losses)
                tf.summary.scalar('cross-entropy',self.loss)
    
            with tf.name_scope('accuracy'):
                self.predictions = tf.cast(tf.argmax(self.D_probs,axis=-1),tf.int32)
                self.accuracy = tf.reduce_mean(tf.cast(tf.less(tf.abs(self.predictions-self.gt),3),tf.float32)) # 3 pixels accuracy
                tf.summary.scalar('acc',self.accuracy)
        

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                            beta1=self.beta1,
                                            beta2=self.beta2)
	tf.summary.scalar('lr',self.lr)
        self.train_op = self.optimizer.minimize(self.loss,global_step=self.global_step)
        self.merged = tf.summary.merge_all()

    def inference(self,sess,img0,img1):
        '''
            return predictions,val_D_probs 
        '''
        feed_dict = {self.val_img0:img0,
                     self.val_img1:img1,
                     self.mode:False}
        return sess.run([self.val_predictions,self.val_D_probs ],feed_dict=feed_dict)

    def test(self,sess):
        '''
            return step,loss,acc,
        '''
        return sess.run([self.global_step,self.loss,self.accuracy])

    def train(self,sess):
        '''
            return  summary,_,step,loss,acc
        '''
        return sess.run([self.merged,self.train_op,self.global_step,self.loss,self.accuracy,self.lr])

    def get_step(self,sess):
        return sess.run(self.global_step)

