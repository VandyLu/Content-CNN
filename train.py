#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cfg
import sys
sys.path.append('./network')
import net as NET

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--ckpt", dest="checkpoint_path", default="", type=str,metavar="FILE", help='model checkpoint path')
args = parser.parse_args()

net = NET.Luo(mode='TRAIN')
with tf.Session() as sess:
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('./log',sess.graph)

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    global_step = 0
    ckpt = tf.train.get_checkpoint_state('./models')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        print 'model restored from:'+ckpt.model_checkpoint_path
        global_step = sess.run(net.global_step)

    for i in range(cfg.param.train_iter):

        runs = [net.merged,
                    net.train_op,
                    net.global_step,
                    net.loss,
                    net.accuracy,
                    net.predictions,
                    net.gt ]
        merged_summary,_,global_step,loss,acc,p,gt = sess.run(runs)
        summary_writer.add_summary(merged_summary,global_step=global_step)
        print 'step:%d\tloss:%.2f\tacc:%.4f\terr:%.1f'%(global_step,loss,acc,np.mean(np.abs(gt-p)))

        if global_step%cfg.param.save_iter == 0:
            save_path = cfg.param.model_save_path.format('Luo',global_step)
            saver.save(sess,save_path)
            print 'model saved to:{}!'.format(save_path)

        if global_step%cfg.param.test_iter==0:
            test_runs = [net.loss,
                    net.accuracy,
                    net.predictions,
                    net.gt ]
            loss,acc,p,gt = sess.run(test_runs,feed_dict={net.mode:False})
            print 'testing:'
            print 'step:%d\tloss:%.2f\tacc:%.4f\terr:%.1f'%(global_step,loss,acc,np.mean(np.abs(gt-p)))
            print np.abs(p-gt)


    coord.request_stop()
    coord.join(threads)



