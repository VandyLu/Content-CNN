#!/usr/bin/python
# -*- coding: utf-8 -*-


class Param():

    def __init__(self):
        self.n_tr = 100
        self.n_te = 100

        self.crop_shape = (360,1200)
        self.window_size = 9
        self.dispmax = 228
        self.kernel = 3
        self.img2batch = 128

        self.dataset = {'TRAIN':'./data/train.tfrecords',
                        'TEST':'./data/val.tfrecords'}
        self.train_iter = 100000
        self.test_iter = 400 # test every 200 iter
        self.learning_rate = 0.00001
        self.beta1 = 0.9
        self.beta2 = 0.99

class Info():

    def __init__(self):
        self.path = '/data/stereo/training/'
        self.disp_path = self.path+'disp_noc_0'
        self.img0_path = self.path+'image2'
        self.img1_path = self.path+'image3'


param = Param()
info = Info() 
