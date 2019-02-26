# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 08:59:40 2019

@author: yiyuezhuo

The loss used here is SSD style but using YOLO style encoding.

The network output encoding, and it will be decoded to compute loss with
target corrds.
"""

import torch



def iou(boxes1, boxes2):
    '''
    batch mode of transforms.iou
    
    boxes1: (num_priors, 4)
    boxes2: (num_boxes, 4)
    
    Return:
        IoU matrix. Shape: (num_priors, num_boxes)
    '''
    xmin1,ymin1,xmax1,ymax1 = boxes1[:,0],boxes1[:,1],boxes1[:,2],boxes1[:,3]
    xmin2,ymin2,xmax2,ymax2 = boxes2[:,0],boxes2[:,1],boxes2[:,2],boxes2[:,3]
    # slice will not copy data, but only generate a proper view of original tensor.
    
    w1,h1 = xmax1-xmin1, ymax1-ymin1
    w2,h2 = xmax2-xmin2, ymax2-ymin2
    
    area1 = w1*h1
    area2 = w2*h2
    
    w = w1 + w2 - (torch.max(xmax1,xmax2) - torch.min(xmin1,xmin2))
    h = h1 + h2 - (torch.max(ymax1,ymax2) - torch.min(ymin1,ymin2))
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
    
    YOLO style decoding([0,1] translate)
    
    b_x = (\sigma(t_x)+c_x)/W
    b_y = (\sigma(t_y)+c_y)/H
    b_w = p_w e^{t_w}/W
    b_h = p_h e^{t_h}/H
    
    t_x,t_y,t_w,t_h are outputs of network.
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

def point_form_loss_matched_with_priors(boxes_p, conf_p, boxes_gt, conf_gt,
                    match_threshold = 0.4):
    raise NotImplementedError
    

def point_form_loss(boxes_p, conf_p, boxes_gt, conf_gt,
                    match_threshold = 0.4):
    '''
    boxes_predict, boxes_gt are point-form tensor/list come from network/ground truth.
    boxes_p: (batch_size, num_priors, 4)
    conf_p: (batch_size, num_priors, num_classes)
    boxes_gt: [[(4),...],...] (length=batch_size)
    conf_gt: [[(1),...],...] (length=batch_size)
    
    Every boxes_p will be mathched with boxes_gt. If one IoU is less than
    '''
    loss_list = []
    batch_size = boxes_p.shape[0]
    for b_p, c_p, b_g_list, c_g_list in zip(boxes_p, conf_p, boxes_gt, conf_gt):
        # For every entries in a batch, is it true to write it in this form?
        b_g = torch.tensor(b_g_list)
        iou_mat = iou(b_p, b_g)
        iou_max_value, iou_max_idx = torch.max(iou_mat, 1)
        mask = iou_max_value > match_threshold
    