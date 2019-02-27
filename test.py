# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:25:58 2019

@author: yiyuezhuo
"""
import torch
from networks import ResNet18Reduced,JanuaryNet
from transforms import test_transform
from losses import decode
from datasets import TestDatasets
from utils import rectangle, show_detection
import torch.nn.functional as F

class config:
    resume = r'weights/Jan_net_epoch=8batch=308.pth'
    datasets = r'E:\agent3\lab\switch\JPEGImages'
    
resnet_features = ResNet18Reduced()
net = JanuaryNet(resnet_features, 3)
net.load_state_dict(torch.load(config.resume))
net.eval()

datasets = TestDatasets(config.datasets)


num_test = 10
for img_numpy in datasets:
    img_tensor = test_transform(img_numpy)
    img_wrap = img_tensor.unsqueeze(0)
    
    loc,conf = net(img_wrap)
    
    conf_prob = F.softmax(conf, dim=2)
    max_value, max_idx = conf_prob[0].max(0)
    print(max_value)
    print(max_idx)
    loc_decoded = decode(loc, net.priors_center_offset)
    h,w,c = img_numpy.shape
    loc_abs = loc_decoded * torch.tensor([w,h,w,h]).float()
    
    print(loc_abs[:,max_idx[0]])
    show_detection(img_numpy, loc_abs[:,max_idx[0]]) # Find max conf
    print(loc_abs[:,max_idx[1]])
    show_detection(img_numpy, loc_abs[:,max_idx[1]]) # Find max conf
    print(loc_abs[:,max_idx[2]])
    show_detection(img_numpy, loc_abs[:,max_idx[2]]) # Find max conf
    
    num_test -= 1
    if num_test <= 0:
        break
    
if False:
    from torchviz import make_dot, make_dot_from_trace
    make_dot(net(img_wrap))