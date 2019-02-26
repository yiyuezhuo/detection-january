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

import matplotlib.patches as patches

from datasets import DetectionDataset
from transforms import random_horizontal_flip, random_crop, photometric_distort, \
    channel_swap, to_tensor, normalize, train_transform, test_transform, \
    enchancement_transform, base_transform,resize

def rectangle(xy, width, height, edgecolor='r', linewidth=1, facecolor='none'):
    rect = patches.Rectangle(xy, width, height, linewidth=linewidth, 
                             edgecolor=edgecolor, facecolor=facecolor)
    plt.gca().add_patch(rect)

def show_detection(img, boxes, show=True):
    plt.imshow(img)
    for box in boxes:
        xmin,ymin,xmax,ymax = box
        rectangle((xmin,ymin), xmax-xmin, ymax-ymin)
    if show:
        plt.show()

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
        print('test random_horizontal_flip')
        for i in range(7):
            ct = c.copy()
            dt = d.copy()
            ct,dt = random_crop(ct, dt)
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
        print('test enchancement_transform')
        for i in range(8):
            ct = c.copy()
            dt = d.copy()
            ct,dt= enchancement_transform(ct, dt)
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