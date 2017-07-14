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
    for i in range(1000):
        gt,_,probs,pre,loss,accuracy = sess.run([net.gt,net.train_op,net.probs,net.predictions,net.loss,net.accuracy])
        print 'iter:{},loss:{},acc:{}'.format(i,loss,accuracy)
        if i%10 ==0:
            print probs
            print np.abs(pre-gt)

    coord.request_stop()
    coord.join(threads)


