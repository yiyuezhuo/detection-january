# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:59:33 2019

@author: yiyuezhuo
"""

import random
import numpy as np

def iou(box1, box2):
    '''
    (xmin1,ymin1,xmax1,ymax1), (xmin2,ymin2,xmax2,ymax2)
    '''
    xmin1,ymin1,xmax1,ymax1 = box1
    xmin2,ymin2,xmax2,ymax2 = box2
    
    w1,h1 = xmax1-xmin1, ymax1-ymin1
    w2,h2 = xmax2-xmin2, ymax2-ymin2
    
    area1 = w1*h1
    area2 = w2*h2
    
    w = w1 + w2 - (max(xmax1,xmax2) - min(xmin1,xmin2))
    h = h1 + h2 - (max(ymax1,ymax2) - min(ymin1,ymin2))
    ai = w*h
    
    return ai/(area1+area2-ai)

def point2centeroffset(point_form):
    # Note the order of xmin,ymin,xmax,ymax does not indicate usage of [xmin:xmax, ymin:ymax]
    # in fact slice operation like above will not show in any place in this project.
    xmin,ymin,xmax,ymax = point_form
    centerx = (xmin+xmax)/2
    centery = (ymin+ymax)/2
    width = xmax - xmin
    height = ymax - ymin
    return centerx,centery,width,height

def random_horizontal_flip(img, boxes):
    '''
    img, boxes -> img,boxes
    
    img is (height, width, channel) from
    box is point-form [(xmin,ymin,xmax,ymax), (xmin,ymin,xmax,ymax), ...]
    '''
    
    height,width, channel = img.shape
    
    if random.random() > 0.5:
        img = img[:,::-1]
        boxes[:,[0,2]] = width - boxes[:,[0,2]]
    
    return img,boxes

def random_crop2(img, boxes):
    height,width, channel = img.shape
    
    if random.random() < 0.2:
        return img,boxes
    if random.random() < 0.2:
        ratio = random.random() * 0.9 + 0.1
    else:
        ratio = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
    
    free_ratio = 1 - ratio
    left_ratio = random.random() * free_ratio 
    top_ratio = random.random() * free_ratio
    
    left = left_ratio * width
    top = top_ratio * height
    
    right = left + width * ratio
    bottom = top + height * ratio
    
    left,top,right,bottom = int(left),int(top),int(right),int(bottom)
    
    boxes_t = []
    for box in boxes:
        xmin,ymin,xmax,ymax = box
        cx = (xmax+xmin)/2
        cy = (ymax+ymin)/2
        if left < cx < right and top < cy < bottom:
            xmin_t = max(xmin - left, 0)
            xmax_t = max(xmax - left, 0)
            ymin_t = max(ymin - top, 0)
            ymax_t = max(ymax - top,0)
            boxes_t.append([xmin_t,ymin_t,xmax_t,ymax_t])
    boxes_t = np.array(boxes_t)
    return img[top:bottom, left:right], boxes_t
    
def random_crop(img, boxes):
    height,width, channel = img.shape
    
    if random.random() < 0.2:
        return img,boxes
    if random.random() < 0.2:
        iou_ratio = 0.0
    else:
        iou_ratio = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
    
    #print('iou_ratio', iou_ratio)
    for i in range(30):
        ratio = 0.1 + random.random()*0.9
        aspect_ratio = random.random()*0.5+0.5
        if random.random() > 0.5:
            aspect_ratio = 1/aspect_ratio
        
        free_ratio = 1 - ratio
        left_ratio = random.random() * free_ratio 
        #top_ratio = random.random() * free_ratio
        top_ratio = left_ratio * aspect_ratio
        
        left = left_ratio * width
        top = top_ratio * height
        
        right = left + width * ratio
        bottom = top + height * ratio
        
        left,top,right,bottom = int(left),int(top),int(right),int(bottom)
        box_alt = [left,top,right,bottom]
        
        if max([iou(box, box_alt) for box in boxes]) > iou_ratio:
            break
    else:
        #print('fail')
        return img, boxes
    
    boxes_t = []
    for box in boxes:
        xmin,ymin,xmax,ymax = box
        cx = (xmax+xmin)/2
        cy = (ymax+ymin)/2
        if left < cx < right and top < cy < bottom:
            xmin_t = max(xmin - left, 0)
            xmax_t = max(xmax - left, 0)
            ymin_t = max(ymin - top, 0)
            ymax_t = max(ymax - top,0)
            boxes_t.append([xmin_t,ymin_t,xmax_t,ymax_t])
    boxes_t = np.array(boxes_t)
    #print('succ', left,top,right,bottom, max([iou(box, box_alt) for box in boxes]) )
    return img[top:bottom, left:right], boxes_t

def photometric_distort(img):
    pass