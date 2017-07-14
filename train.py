#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cfg
import sys
sys.path.append('./network')
import net as NET

net = NET.Luo(mode='TRAIN')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(cfg.param.train_iter):
        all,gt,_,probs,pre,loss,accuracy = sess.run([net.all_probs,net.gt,net.train_op,net.probs,net.predictions,net.loss,net.accuracy])
        print 'iter:{},loss:{},err:{}'.format(i,loss,np.mean(np.abs(gt-pre)))
        if i%cfg.param.test_iter==0:
            print probs
            print np.abs(pre-gt)

    coord.request_stop()
    coord.join(threads)



