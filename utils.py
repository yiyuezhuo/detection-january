# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 08:56:17 2019

@author: yiyuezhuo
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch.nn.functional as F
import torch

# plot uitls
def rectangle(xy, width, height, edgecolor='r', linewidth=1, facecolor='none'):
    rect = patches.Rectangle(xy, width, height, linewidth=linewidth, 
                             edgecolor=edgecolor, facecolor=facecolor)
    plt.gca().add_patch(rect)
    

def show_detection(img, boxes, show=True, text_list = None,
                   fontsize = 12):
    plt.imshow(img)
    for i,box in enumerate(boxes):
        xmin,ymin,xmax,ymax = box
        #print((xmin,ymin), xmax-xmin, ymax-ymin)
        rectangle((xmin,ymin), xmax-xmin, ymax-ymin)
        if text_list is not None:
            plt.text(xmin, ymin-30, text_list[i], fontsize=fontsize, bbox=dict(facecolor='purple', alpha=0.1))
    if show:
        plt.show()

# computing utils
def iou(boxes1, boxes2):
    '''
    batch mode of transforms.iou
    
    boxes1: (num_priors, 4)
    boxes2: (num_boxes, 4)
    
    Return:
        IoU matrix. Shape: (num_priors, num_boxes)
    '''
    #num_priors = boxes1.shape[0]
    #num_boxes = boxes2.shape[0]
    
    xmin1,ymin1,xmax1,ymax1 = boxes1[:,0],boxes1[:,1],boxes1[:,2],boxes1[:,3]
    xmin2,ymin2,xmax2,ymax2 = boxes2[:,0],boxes2[:,1],boxes2[:,2],boxes2[:,3]
    # slice will not copy data, but only generate a proper view of original tensor.
    
    xmin1,xmax1,ymin1,ymax1 = xmin1.view(-1,1),xmax1.view(-1,1),ymin1.view(-1,1),ymax1.view(-1,1)
    xmin2,xmax2,ymin2,ymax2 = xmin2.view(1,-1),xmax2.view(1,-1),ymin2.view(1,-1),ymax2.view(1,-1)
    
    w1,h1 = xmax1-xmin1, ymax1-ymin1
    w2,h2 = xmax2-xmin2, ymax2-ymin2
    
    area1 = w1*h1
    area2 = w2*h2
    
    # For torch.tensor(0) get on cuda device defaultly, we need lines such as 
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    w = torch.max(w1 + w2 - (torch.max(xmax1,xmax2) - torch.min(xmin1,xmin2)), torch.tensor(0.0))
    h = torch.max(h1 + h2 - (torch.max(ymax1,ymax2) - torch.min(ymin1,ymin2)), torch.tensor(0.0))
    ai = w*h
    
    return ai/(area1+area2-ai)

        
def nms(boxes, conf_prob, conf_threshold =0.5, iou_threshold =0.45):
    '''
    Non-Maximum Suppression
    boxes, conf are decoded network output except batch_size dimention.
    
    boxes: (num_priors, 4): point-form ([0,1] is not required, but valid)
    conf_prob: (num_priors, n_classes) [0,1] softmaxed, including 0 as background class
    Return:
        idx_list 
        # Max class will be determined by caller.
    '''
    n_classes = conf_prob.shape[1]
    
    #conf_prob = F.softmax(conf, 1) # (num_priors, n_classes)
    
    keep = []
    for cls_idx in range(1, n_classes):#without background class
        mask_cls = conf_prob[:, cls_idx] > conf_threshold
        idxs = torch.nonzero(mask_cls).view(-1) #(num_priors') index
        
        while len(idxs)>0:
            conf_prob_masked = conf_prob[idxs,:] #(num_priors') bool
            conf_prob_masked_sv, conf_prob_masked_si = conf_prob_masked[:, cls_idx].sort(0,descending=True)
    
            idxs = idxs[conf_prob_masked_si]
            keep.append(idxs[0])
            
            ratio = iou(boxes[[idxs[0]],:] ,boxes[idxs,:])[0]
            mask_iou = ratio < iou_threshold
            idxs = idxs[mask_iou]
    
    if len(keep) == 0:
        # workaround for: RuntimeError: expected a non-empty list of Tensors
        return torch.tensor([]).long()
    return torch.stack(keep, 0)

def mAP():
    pass

