# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:04:06 2019

@author: yiyuezhuo
"""
#%load_ext autoreload
#%autoreload 2


import matplotlib.pyplot as plt
import numpy as np
#import imageio


from datasets import DetectionDataset
from transforms import random_horizontal_flip, random_crop, photometric_distort, \
    channel_swap, to_tensor, normalize, train_transform, test_transform, \
    enchancement_transform, base_transform,resize
from utils import rectangle, show_detection

if __name__ == '__main__':
    dataset = DetectionDataset(r'E:\agent3\lab\switch', ['open', 'close'])
    for c,d,e in dataset:
        break
    
    if True:
        print('test random_horizontal_flip')
        for i in range(5):
            ct = c.copy()
            dt = d.copy()
            ct,dt = random_horizontal_flip(ct, dt)
            show_detection(ct, dt)
    
    if True:
        print('test random_crop')
        for i in range(7):
            ct = c.copy()
            dt = d.copy()
            ce = e.copy()
            ct,dt,ce = random_crop(ct, dt, ce)
            show_detection(ct, dt)
            
    if True:
        print('test photometric_distort')
        for i in range(5):
            ct = c.copy()
            dt = d.copy()
            ct= photometric_distort(ct)
            show_detection(ct, dt)
    
    if True:
        print('test channel_swap')
        for i in range(5):
            ct = c.copy()
            dt = d.copy()
            ct= channel_swap(ct)
            show_detection(ct, dt)
            
    if True:
        print('test enchancement_transform(image 0)')
        for i in range(5):
            ct = c.copy()
            dt = d.copy()
            ce = e.copy()
            ct,dt,ce= enchancement_transform(ct, dt, ce)
            #if len(dt.squeeze()) != 0:
            if dt.numel() != 0:
                dt[:,[0,2]] *= ct.shape[1]
                dt[:,[1,3]] *= ct.shape[0]
            show_detection(ct, dt)
            
    if True:
        print('test resize')
        for i in range(1):
            ct = c.copy()
            dt = d.copy()
            ct= resize(ct)
            show_detection(ct, dt)
            
    if True:
        print('test enchancement_transform(many image)')
        num_test = 10
        for c,d,e in dataset:
            ct = c.copy()
            dt = d.copy()
            ct,dt,ce= enchancement_transform(ct, dt, ce)
            
            print(dt)
            #if len(dt.squeeze()) != 0:
            if dt.numel() !=0:
                dt[:,[0,2]] *= ct.shape[1]
                dt[:,[1,3]] *= ct.shape[0]
            show_detection(ct, dt)
            
            
            num_test -=1
            if num_test<=0:
                break

    
