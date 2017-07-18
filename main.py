#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time 
import matplotlib.pyplot as plt
import cv2
import os
import cfg
import sys
sys.path.append('./network')
sys.path.append('./lib')
import Content_CNN as NET
from StereoInput import val_pipeline 
import io_disp_tools
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='1'

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", dest="mode", default="train", type=str, help='mode:train or val')
parser.add_argument("-c", "--ckpt", dest="checkpoint_path", default="", type=str,metavar="FILE", help='model checkpoint path')
parser.add_argument("-s","--save",dest="save_path",default="",type=str,help='model save path')
parser.add_argument("-l","--log",dest="log_path",default="./log",type=str,help='model save path')
args = parser.parse_args()

np.set_printoptions(threshold='nan')#print all
net = NET.Content_CNN()

if not os.path.isdir(args.log_path):
    os.mkdir(args.log_path)

with tf.Session() as sess:
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(args.log_path,sess.graph)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    if not args.checkpoint_path == "":
        ckpt = tf.train.get_checkpoint_state(args.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            print 'Model restored from:'+ckpt.model_checkpoint_path

    if args.mode == 'train':
        for _ in range(cfg.train_iter):
            summary,_,step,loss,acc = net.train(sess)
            summary_writer.add_summary(summary,global_step=step)

            if step % cfg.display_iter == 0:
                print 'step:%d\tloss:%.2f\tacc:%.4f'%(step,loss,acc)

            if step % cfg.save_iter == 0:
                if not args.save_path == "":
                    model_iter_path = os.path.join(args.save_path,'Luo_{}.ckpt'.format(step))
                    saver.save(sess,model_iter_path)
                    print 'Model saved to {}!'.format(model_iter_path)

            if step % cfg.test_iter == 0:
                step,loss,acc = net.test(sess)
                print 'test:'
                print 'step:%d\tloss:%.2f\tacc:%.4f'%(step,loss,acc)

    if args.mode == 'val':
        batch_size = 36
        n = 3
        for j in range(12):
            img0,img1 = val_pipeline(n) # numpy ndarray (128,H,W,C)
        
            t0 = time.time()
            predictions,D_probs = net.inference(sess,img0,img1)
            t1 = time.time()
            print 'time:{}s/img'.format((t1-t0)/n)

            predictions = predictions.astype(np.float)
            for i in range(n):
                disp = io_disp_tools.dispFromArray(predictions[i])
                disp.writeColor('output/disp/{}.png'.format(i+j*n),110)
                cv2.imwrite('output/image_2/{}.png'.format(i+j*n),img0[i])
                cv2.imwrite('output/image_3/{}.png'.format(i+j*n),img1[i])
    


    coord.request_stop()
    coord.join(threads)


