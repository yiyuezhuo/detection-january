# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:09:24 2019

@author: yiyuezhuo
"""

from utils import nms, show_detection
from datasets import TestDatasets
import torch

class config:
    datasets = r'E:\agent3\lab\switch\JPEGImages'


boxes_base = torch.tensor([[0.1,0.1,0.3,0.3], [0.7,0.7,0.9,0.9]])
boxes_noised = boxes_base.repeat(1,10).view(-1,4)
boxes_noised = boxes_noised + torch.randn_like(boxes_noised) * 0.02

datasets = TestDatasets(config.datasets)
for img in datasets:
    break

print('The image is only selected for test, boxes are fake input instead of output of model.')
h,w,c = img.shape
print('before nms')
show_detection(img, boxes_noised * torch.tensor([w,h,w,h]).float())

print('after nms')
fake_conf = torch.zeros(boxes_noised.shape[0], 3)
fake_conf[:10,1] = 10.
fake_conf[10:,2] = 10.
boxes_idx = nms(boxes_noised, fake_conf)
boxes_nmsed = boxes_noised[boxes_idx, :]
show_detection(img, boxes_nmsed * torch.tensor([w,h,w,h]).float())

print('corner case test(one class miss)')
fake_conf = torch.zeros(boxes_noised.shape[0], 3)
fake_conf[:10,1] = 10.
#fake_conf[10:,2] = 10.
boxes_idx = nms(boxes_noised, fake_conf)
boxes_nmsed = boxes_noised[boxes_idx, :]
show_detection(img, boxes_nmsed * torch.tensor([w,h,w,h]).float())

print('corner case test(only background class)')
fake_conf = torch.zeros(boxes_noised.shape[0], 3)
#fake_conf[:10,1] = 10.
#fake_conf[10:,2] = 10.
boxes_idx = nms(boxes_noised, fake_conf)
boxes_nmsed = boxes_noised[boxes_idx, :]
show_detection(img, boxes_nmsed * torch.tensor([w,h,w,h]).float())
