# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:26:19 2019

@author: yiyuezhuo
"""

from detect import DetectionAlgorithm
from utils import show_detection
import imageio
import glob
import os
import matplotlib.pyplot as plt
from datasets import DetectionTestDataset

print('start setup detector')
detector = DetectionAlgorithm('weights/Jan_net_epoch=25batch=322.pth', 
                              ['open', 'close'],
                              conf_threshold = 0.95,
                              iou_threshold = 0.01)

print('end setup detector')
# detect, single image detect API test
if True:
    img = imageio.imread(r'E:\agent3\lab\switch\JPEGImages\1001.jpg')
    det = detector.detect(r'E:\agent3\lab\switch\JPEGImages\1001.jpg')
    show_detection(img, det['absolute_box'])

# test batch convert(cpu)
if False:
    for fname in glob.glob('E:/agent3/lab/switch2/JPEGImages/*.jpg'):
        img = imageio.imread(fname)
        det = detector.detect(img)
        
        name, _ = os.path.splitext(os.path.split(fname)[-1])
        target_path = os.path.join('images/eval', name+'.jpg')
        
        print('{} -> {}'.format(fname, target_path))
        
        show_detection(img, det['absolute_box'],text_list=det['class'], show=False)
        plt.savefig(target_path)
        plt.clf()    

# test setuping mAP 
if False:
    detector.mAP_setup(r'E:\agent3\lab\switch', target=r'E:\agent4\mAP\predicted' ,idx_file='test')

if False:
    datasets = DetectionTestDataset(r'E:\agent3\lab\switch',['open','close'], idx_file='test')
    datasets.copy_annotations(r'E:\agent4\mAP\ground-truth')