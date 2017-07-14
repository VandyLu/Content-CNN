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
        runs = [    net.merged,
                    net.train_op,
                    net.global_step,
                    net.loss,
                    net.accuracy ]

        if global_step%cfg.param.test_iter==0:
            #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()
            merged_summary,_,global_step,loss,acc = sess.run(runs)
            #        options=run_options,run_metadata=run_metadata)

            #summary_writer.add_run_metadata(run_metadata,'step{}'.format(global_step))
            summary_writer.add_summary(merged_summary,global_step=global_step)
        else:
            merged_summary,_,global_step,loss,acc = sess.run(runs)
            summary_writer.add_summary(merged_summary,global_step=global_step)
        print 'step:{},loss:{},acc:{}'.format(global_step,loss,acc)

        if global_step%cfg.param.save_iter == 0:
            save_path = cfg.param.model_save_path.format('Luo',global_step)
            saver.save(sess,save_path)
            print 'model saved to:{}!'.format(save_path)

    coord.request_stop()
    coord.join(threads)



