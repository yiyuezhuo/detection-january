# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:33:40 2019

@author: yiyuezhuo
"""

from datasets import SwitchDatasets
import torch
from torch.utils import data
from networks import ResNet18Reduced, JanuaryNet, weight_init
from losses import decode, point_form_loss

class config:
    data_root = r'E:\agent3\lab\switch'
    batch_size = 8
    shuffle = True
    #num_workers = 0 # For debug purpose: https://github.com/pytorch/pytorch/issues/2341
    num_workers = 2 # each worker load a single batch and return it once it's ready
    pin_memory = False # preload content into VRAM
    
    num_epoch = 100

def collate_fn(batch):
    img_list = []
    coords_list = []
    labels_list = []
    for img, coords, labels in batch:
        img_list.append(img)
        coords_list.append(coords)
        labels_list.append(labels)
    return torch.stack(img_list, 0), coords_list, labels_list

if __name__ == '__main__':
    print('config:')
    for key,value in vars(config).items():
        if not key.startswith('__'):
            print('{} = {}'.format(key, value))

    datasets = SwitchDatasets(config.data_root)
    
    dataloader = data.DataLoader(datasets, 
                    batch_size = config.batch_size,
                    shuffle = config.shuffle,
                    num_workers = config.num_workers,
                    pin_memory = config.pin_memory,
                    collate_fn = collate_fn)
    
    for epoch_idx in range(config.num_epoch):
        for batch_idx,(imgs, coords_list, labels_list) in enumerate(dataloader):
            break
        break
    
    samples = []
    for i,sample in enumerate(datasets):
        if i==8:
            break
        samples.append(sample)
    imgs, coords_list, labels_list =  collate_fn(samples) # coords_list is point-form
    
    # setup backbone network
    resnet_features = ResNet18Reduced()
    resnet_features.load_state_dict(torch.load('weights/resnet18reduced.pth'))
    
    # setup main network
    net = JanuaryNet(resnet_features, 3)
    net.init(weight_init)
    
    loc,conf = net(imgs) # loc is encoding, 
    loc_decoded = decode(loc) # (batch_size, num_priors, 4) point-form
    loss = point_form_loss(loc_decoded, conf, coords_list, labels_list)
