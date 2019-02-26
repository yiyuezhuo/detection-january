# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 08:59:40 2019

@author: yiyuezhuo

The loss used here is SSD style but using YOLO style encoding.

The network output encoding, and it will be decoded to compute loss with
target corrds.
"""

import torch
import torch.nn.functional as F


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

def encode():
    '''
    This implementation don't require encode
    '''
    raise NotImplementedError

def decode(boxes, priors):
    '''
    (batch_size, num_priors, 4), (num_priors, 4) -> (batch_size, num_priors, 4)
    boxes: Output of network
    priors: center-offset form priors boxes
    
    YOLO style decoding([0,1] translate)
    
    b_x = (\sigma(t_x)+c_x)/W
    b_y = (\sigma(t_y)+c_y)/H
    b_w = p_w e^{t_w}/W
    b_h = p_h e^{t_h}/H
    
    t_x,t_y,t_w,t_h are outputs of network.(t means translate)
    b_x,b_y,b_w,b_h are center-offset form of bboxes.
    
    Return:
        point-form output. shape: (batch_size, num_priors, 4)
    '''
    t_x, t_y, t_w, t_h = boxes[:,:,0],boxes[:,:,1],boxes[:,:,2],boxes[:,:,3]
    c_x, c_y, p_w, p_h = priors[:,0], priors[:,1], priors[:,2], priors[:,3]
    
    b_x = torch.sigmoid(t_x) + c_x #(batch_size, num_priors)
    b_y = torch.sigmoid(t_y) + c_y
    b_w = p_w * torch.exp(t_w)
    b_h = p_h * torch.exp(t_h)
    
    return torch.stack([b_x-b_w/2, b_y-b_h/2, b_x+b_w/2, b_y+b_h/2], 2)

'''
def point_form_loss_matched_with_priors(boxes_p, conf_p, boxes_gt, conf_gt, priors,
                    match_threshold = 0.4):
    raise NotImplementedError
'''

def point_form_loss(boxes_p, conf_p, boxes_gt, conf_gt, priors,
                    match_threshold = 0.5, negative_odd = 3., verbose=True):
    '''
    boxes_predict, boxes_gt are point-form tensor/list come from network/ground truth.
    
    boxes_p: (batch_size, num_priors, 4)
    conf_p: (batch_size, num_priors, num_classes)
    boxes_gt: [[(4),...],...] (length=batch_size)
    conf_gt: [[(1),...],...] (length=batch_size)
    priors: point-form priors (num_priors, 4)
    
    Every boxes_p will be mathched with boxes_gt. If one IoU is less than
    '''
    batch_size = boxes_p.shape[0]    
    
    loc_losses = torch.zeros(batch_size)
    conf_losses = torch.zeros(batch_size)
    for i, (b_p, c_p, b_g_list, c_g_list) in enumerate(zip(boxes_p, conf_p, boxes_gt, conf_gt)):
        # For every entries in a batch, is it true to write it in this form?
        if len(b_g_list.squeeze()) == 0:
            continue # skip if no ground truth box , 
            # Yes, we will not use the background in here to train, considering the requirement of Hard Negative Mining.
        
        b_g = b_g_list.type(torch.float) # (num_gt, 4) 
        #iou_mat = iou(b_p, b_g) # (num_priors, num_gt)
        iou_mat = iou(priors, b_g) # use priors insted of b_p to ensure strong matching
        iou_max_value, iou_max_idx = torch.max(iou_mat, 1) # (num_priors)
        mask_p = iou_max_value > match_threshold # (num_priors) bool
        
        if mask_p.sum() == 0:
            if verbose:
                print("Prior box grid fail to match a given bbox. Maybe you need redesign network or redefine the transforms.")
            continue
        
        b_p_masked = b_p[mask_p,:]
        b_g_selected = torch.index_select(b_g, 0, iou_max_idx)
        b_g_selected = b_g_selected[mask_p,:]
        
        # Localization loss
        loc_losses[i] = F.smooth_l1_loss(b_p_masked, b_g_selected)
        
        # Confidence loss
        # network will output [0, num_class](0 denote background) For switch data, 
        # it will be{0,1,2}. But the conf_gt is [0, num_class-1](0 denotes smoe class
        # of object).
        
        '''
        # Confidence loss without Hard Negative Mining
        c_g = c_g_list + 1 #(num_gt) int
        c_g_selected = torch.index_select(c_g, 0, iou_max_idx)
        c_g_selected[~mask] = 0 # set background class
        
        conf_losses[i] = F.cross_entropy(c_p, c_g_selected.long()) # Why int32 int64 does matter???        
        '''
        
        # Confidence loss with Hard Negative Mining
        c_g = c_g_list + 1 #(num_gt) int
        c_g_selected = torch.index_select(c_g, 0, iou_max_idx)
        c_g_selected[~mask_p] = 0 # set background class
        
        num_negative = (mask_p.sum().float() * negative_odd).floor().long()
        iou_max_value_n = iou_max_value[~mask_p]
        _, negative_idx = iou_max_value_n.sort(descending=True)
        _, negative_rank = negative_idx.sort()
        mask_n = negative_rank < num_negative
                
        loss_p = F.cross_entropy(c_p[mask_p,:], c_g_selected[mask_p].long())
        loss_n = F.cross_entropy(c_p[~mask_p][mask_n,:], c_g_selected[~mask_p][mask_n].long())
        
        conf_losses[i] = loss_p + loss_n 
        # It's not equivalent to [mask_p | mask_n,:] since cross_entropy control size.
    
    loc_loss = loc_losses.sum() 
    conf_loss = conf_losses.sum() 
    #loss = loc_loss + alpha * conf_loss
    
    return loc_loss, conf_loss
    
#point_form_loss = point_form_loss_matched_with_priors