#!/usr/bin/python
# -*- coding: utf-8 -*-

model_name = 'Content_CNN'

# dataset prepare
n_tr = 100
n_te = 100
crop_shape = (360,1200)
dataset = {'TRAIN':'./data/train.tfrecords',
                'TEST':'./data/val.tfrecords'}

# origin dataset path
path = '/data/stereo/training/'
disp_path = path+'disp_noc_0'
img0_path = path+'image_2'
img1_path = path+'image_3'
# val set
val_path = './data/data_scene_flow/testing/'
val_img0 = val_path+'image_2/'
val_img1 = val_path+'image_3/'

# network structure
window_size = 9
dispmax = 228
kernel = 3
batch_size = 256
img_num = 4

# trainig params
train_iter = 500000
test_iter = 100 # test every 200 iter
save_iter = 500
display_iter = 20
learning_rate = 0.0000008
decay_steps = 8000
decay_rate = 1.0
beta1 = 0.9
beta2 = 0.99

