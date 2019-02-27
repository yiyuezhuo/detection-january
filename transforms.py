# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:59:33 2019

@author: yiyuezhuo
"""

import random
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv # I hate OpenCV anyway -_-
from torchvision.transforms import ToTensor,Normalize
import torch

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
    
    w = max(w1 + w2 - (max(xmax1,xmax2) - min(xmin1,xmin2)), 0)
    h = max(h1 + h2 - (max(ymax1,ymax2) - min(ymin1,ymin2)), 0)
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

def centeroffset2point(centeroffset_form):
    centerx,centery,width,height = centeroffset_form
    xmin = centerx - width/2
    xmax = centerx + width/2
    ymin = centery - height/2
    ymax = centery + height/2
    return xmin,ymin,xmax,ymax

def random_horizontal_flip(img, boxes):
    '''
    img, boxes -> img,boxes
    
    img is (height, width, channel) from
    box is point-form [(xmin,ymin,xmax,ymax), (xmin,ymin,xmax,ymax), ...]
    '''
    
    height,width, channel = img.shape
    non_boxes = len(boxes.squeeze()) == 0
    #non_boxes = boxes.numel() == 0 # well, this is numpy array
    
    if not non_boxes and random.random() > 0.5:
        img = img[:,::-1]
        boxes[:,[0,2]] = width - boxes[:,[2,0]]
    
    return img,boxes

def random_crop_wrong(img, boxes):
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
    
def random_crop(img, boxes, labels):
    height,width, channel = img.shape
    
    if random.random() < 0.2:
        return img,boxes,labels
    if random.random() < 0.2:
        iou_ratio = 0.0
    else:
        iou_ratio = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
    
    #print('iou_ratio', iou_ratio)
    non_boxes = len(boxes.squeeze()) == 0
    for i in range(30):
        aspect_ratio = random.random()*0.5+0.5

        x_ratio = 0.1 + random.random()*0.9
        y_ratio = x_ratio * aspect_ratio # it's only true if origin image is squre
        if random.random() > 0.5:
            y_ratio,x_ratio = x_ratio,y_ratio
        
        #free_ratio = 1 - ratio
        left_ratio = random.random() * (1 - x_ratio) 
        top_ratio = random.random() * (1 - y_ratio) 
                
        left = left_ratio * width
        top = top_ratio * height
        
        right = left + width * x_ratio
        bottom = top + height * y_ratio
        
        left,top,right,bottom = int(left),int(top),int(right),int(bottom)
        box_alt = [left,top,right,bottom]
        
        if non_boxes or max([iou(box, box_alt) for box in boxes]) > iou_ratio:
            break
    else:
        #print('fail')
        return img, boxes, labels
    
    #print(left,top,right,bottom)
    
    boxes_t = []
    labels_t = []
    for label, box in zip(labels, boxes):
        xmin,ymin,xmax,ymax = box
        cx = (xmax+xmin)/2
        cy = (ymax+ymin)/2
        if left < cx < right and top < cy < bottom:
            xmin_t = max(xmin - left, 0)
            xmax_t = max(xmax - left, 0)
            ymin_t = max(ymin - top, 0)
            ymax_t = max(ymax - top,0)
            
            boxes_t.append([xmin_t,ymin_t,xmax_t,ymax_t])
            labels_t.append(label)
    
    boxes_t = np.array(boxes_t)
    labels_t = np.array(labels_t)
    #print('succ', left,top,right,bottom, max([iou(box, box_alt) for box in boxes]) )
    #print(img[top:bottom, left:right].shape)
    return img[top:bottom, left:right], boxes_t, labels_t

def photometric_distort_scipy(img):
    '''
    Apply multiplication noise on hsv space.
    '''
    img = img / 255 # hsv_to_rgb take [0,1] rgb value.
    # TODO: rgb_to_hsv is too slow, maybe use of opencv is prefered. 
    hsv = rgb_to_hsv(img)
    hsv[:,:,0] = np.clip(hsv[:,:,0] * (random.random()+0.5), 0.0, 1.0)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (random.random()+0.5), 0.0, 1.0)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * (random.random()+0.5), 0.0, 1.0)
    rgb = hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)

def photometric_distort_opencv(img):
    '''
    Apply multiplication noise on hsv space.(Opencv)
    
    Opencv version:
        63.5 ms ± 1.14 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    scipy version:
        893 ms ± 5.73 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    
    It's too bad to see how poor the scipy is...
    '''
    #import cv2
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,0] = np.clip(hsv[:,:,0] * (random.random()+0.5), 0, 255)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (random.random()+0.5), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * (random.random()+0.5), 0, 255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def resize_PIL(img):
    from PIL import Image
    return Image.fromarray(img).resize((400,400))

def resize_skimage(img):
    from skimage.transform import resize
    return resize(img, (400, 400))

def resize_opencv(img):
    '''
    Opencv version:
        625 µs ± 16.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    PIL version(old scipy refer to):
        4.94 ms ± 276 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    skimage version:
        109 ms ± 505 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    
    coords should be converted to percent value before it. 
    '''
    return cv2.resize(img, (400, 400))

def box_to_percent(img, boxes):
    '''
    Convert coords of boxes to (0,1) range
    
    This function should be called before resizing of image
    '''
    non_boxes = len(boxes.squeeze()) == 0
    if non_boxes:
        return img, boxes
    
    height,width, channel = img.shape
    boxes = boxes.astype(float)
    boxes[:,[0,2]] = boxes[:,[0,2]] / width
    boxes[:,[1,3]] = boxes[:,[1,3]] / height
    return img, boxes
    
try:
    import cv2
    photometric_distort = photometric_distort_opencv
    resize = resize_opencv
except:
    print("cv2 load fail, use alternative version instead. It may be much slow than opencv version")
    photometric_distort = photometric_distort_scipy
    resize = resize_PIL

def channel_swap(img):
    if random.random() > 0.5:
        return img
    perm = random.choice([(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)])
    img[:,:,[0,1,2]] = img[:,:,perm]
    return img

to_tensor = ToTensor()
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# compiled transforms
def enchancement_transform(img, boxes, labels):
    # boxes, labels related operation
    img, boxes, labels = random_crop(img, boxes, labels) # It may reduce size of boxes and labels
    img, boxes = random_horizontal_flip(img, boxes)
    img, boxes = box_to_percent(img, boxes)
    # point operation
    img = photometric_distort(img)
    img = channel_swap(img) # Is it useful?
    return img, boxes, labels

def base_transform(img):
    img = resize(img)
    img = to_tensor(img)
    img = normalize(img)
    return img

def train_transform(img, boxes, labels):
    img, boxes, labels = enchancement_transform(img, boxes, labels)
    img = base_transform(img)
    return img, torch.from_numpy(boxes), torch.from_numpy(labels)

def test_transform(img):
    img = base_transform(img)
    return img
    