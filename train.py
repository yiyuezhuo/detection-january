# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:33:40 2019

@author: yiyuezhuo
"""

import torch
from torch.utils import data
import torch.optim as optim
import os
import time

from datasets import SwitchDatasets#, collate_fn
from networks import ResNet18Reduced, JanuaryNet, weight_init
from losses import decode, point_form_loss


class config:
    # dataset dataloader parameters
    # Fake args object. Maybe it will be replace with a instance of argparse
    data_root = r'E:\agent3\lab\switch'
    batch_size = 8
    shuffle = True
    num_workers = 2 # For debug purpose: https://github.com/pytorch/pytorch/issues/2341
    #num_workers = 2 # each worker load a single batch and return it once it's ready
    pin_memory = False # preload content into VRAM for speed up. It may cause system freeze or something.
    # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
    
    # cache parameters
    #resume = None # resume path, if None create a new network
    resume = 'weights/Jan_net_epoch=8batch=275.pth'
    
    # cuda parameters 
    cuda = True
    
    # trainer parameter
    num_epoch = 100
    
    # loss parameter 
    match_threshold = 0.4
    alpha = 1.0
    
    # optimizer parameter
    #lr = 0.000001
    lr=1e-10
    momentum = 0.9
    #momentum = 0.0
    weight_decay = 5e-4
    #weight_decay = 0.0
    
    # Inspecting parameter
    num_batch_display = 10
    time_to_save = 1800 # second



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
                    collate_fn = SwitchDatasets.collate_fn)
    
    '''
    samples = []
    for i,sample in enumerate(datasets):
        if i==8:
            break
        samples.append(sample)
    imgs, coords_list, labels_list =  collate_fn(samples) # coords_list is point-form
    '''
    
    if config.resume is None:
        # setup backbone network
        print('create new model')
        resnet_features = ResNet18Reduced()
        resnet_features.load_state_dict(torch.load('weights/resnet18reduced.pth'))
        
        # setup main network
        net = JanuaryNet(resnet_features, 3)
        net.init(weight_init)
    else:
        print('load {}'.format(config.resume))
        resnet_features = ResNet18Reduced()
        net = JanuaryNet(resnet_features, 3)
        net.load_state_dict(torch.load(config.resume))
    
    if config.cuda:
        net = net.cuda()
        net.priors_center_offset = net.priors_center_offset.cuda()
        net.priors_point_form = net.priors_point_form.cuda()
        
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # So free constant, such as torch.tensor(0) will be placed on cuda automatically
    
    optimizer = optim.SGD(net.parameters(), 
                  lr = config.lr, momentum = config.momentum, 
                  weight_decay = config.weight_decay)
    
    time_acc = 0
    batch_acc = 0
    loc_loss_acc = 0
    conf_loss_acc = 0
    last_time = time.time()
    for epoch_idx in range(config.num_epoch):
        for batch_idx,(imgs, coords_list, labels_list) in enumerate(dataloader):
            
            if config.cuda: # pin_memory=False
                imgs = imgs.cuda()
                coords_list = [coords.cuda() for coords in coords_list]
                labels_list = [labels.cuda() for labels in labels_list] 
            
            loc,conf = net(imgs) # loc is encoding, 
            loc_decoded = decode(loc, net.priors_center_offset) # (batch_size, num_priors, 4) point-form
            loc_loss, conf_loss = point_form_loss(loc_decoded, conf, coords_list, 
                  labels_list, net.priors_point_form, 
                  match_threshold = config.match_threshold, verbose=False)
            loss = loc_loss + config.alpha*conf_loss 
            
            loss.backward()
            
            optimizer.step()
            
            # Inspecting
            
            now_time = time.time()
            time_acc += now_time - last_time
            last_time = now_time
            batch_acc += 1
            loc_loss_acc += loc_loss.item()
            conf_loss_acc += conf_loss.item()
            
            #print('testing')
            
            if time_acc > config.time_to_save:
                name = 'Jan_net_epoch={}batch={}.pth'.format(epoch_idx, batch_idx)
                path = os.path.join('weights', name)
                torch.save(net.state_dict(), path)
                print('save -> {}'.format(path))
                time_acc = 0
            
            if batch_acc > config.num_batch_display:
                info = 'epoch={} batch={} loc_loss={} conf_loss={}'.format(
                    epoch_idx, batch_idx, loc_loss_acc, conf_loss_acc)
                print(info)
                with open('log.txt', 'a') as f:
                    f.write(info)
                    f.write('\n')
                
                batch_acc = 0
                loc_loss_acc = 0
                conf_loss_acc = 0

            
            #break # Instant exit for debugging
        #break # Instance exit for debugging

    
