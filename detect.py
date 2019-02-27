# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:42:46 2019

@author: yiyuezhuo
"""

'''
Network can only output loc encoding and raw conf score. The decoding, threshold
and nms are required to construct a whole algorithm that can be evaluated
by index such as mAP. Those post-procedure are wraped in this module.
'''

import torch
import torch.nn.functional as F
from torch.utils import data
import imageio
import os


from networks import ResNet18Reduced,JanuaryNet
from utils import nms
from transforms import test_transform
from losses import decode
from datasets import DetectionTestDataset



class config:
    resume = r'weights/Jan_net_epoch=25batch=322.pth'
    batch_size = 8
    num_workers = 0


class DetectionAlgorithm:
    def __init__(self, resume, cls_list,
                 conf_threshold = 0.5, iou_threshold=0.45):
        '''
        resume: Path of weight of model
        cls_list: list of classes. E.g ['open', 'close'] for switch dataset.
        '''
        resnet_features = ResNet18Reduced()
        net = JanuaryNet(resnet_features, 3)
        net.load_state_dict(torch.load(resume))
        net.eval()
        
        self.net = net
        self.cls_list = cls_list
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    def detect(self, path_or_img):
        '''
        Detect single image. path_or_img is image path or raw numpy image.
        '''
        if isinstance(path_or_img, str):
            img = imageio.imread(path_or_img)
        else:
            img = path_or_img
        
        return self.detect_batch([img])[0]
        
    def detect_batch(self, imgs, path_list=None):
        '''
        imgs: list of numpy images. Their shape may have different values.
        '''
        if path_list is None:
            path_list = [None] * len(imgs)
        
        img_list = [test_transform(img) for img in imgs] 
        # test_transform: (height, width, 3) -> (3, height_resized, width_resized)
        img_wrap = torch.stack(img_list, 0)
        meta_list = [{'height':img.shape[0],
                      'width': img.shape[1],
                      'path': p} for img,p in zip(imgs,path_list)]
        
        return self._detect_batch(img_wrap, meta_list)
    def _detect_batch(self, imgs, meta_list):
        '''
        imgs, meta_list are items of dataloader of DetectionTestDataset
        So imgs have been transformed and meta_list contain information such
        as original width and height of images.
        
        imgs: (batch_size, 3, height_net, width_net)
        meta_list: [{height_ori:..., width_ori:...},...]
        '''
        loc,conf = self.net(imgs)
        loc_decoded = decode(loc, self.net.priors_center_offset) # [0,1] values
        conf_prob = F.softmax(conf, dim=2)
        
        output_list = []
        for meta, loc_decoded_v, conf_prob_v in zip(meta_list, loc_decoded, conf_prob):
            h,w = meta['height'], meta['width']
            
            idxs = nms(loc_decoded_v, conf_prob_v, conf_threshold = self.conf_threshold,
                       iou_threshold = self.iou_threshold)
            
            if idxs.numel() == 0:
                out = {'percent_box': torch.tensor([]),
                       'absolute_box': torch.tensor([]),
                       'confidence': torch.tensor([]),
                       'class': [],
                       'width': w,
                       'height': h}
                output_list.append(out)
                continue
            
            loc_decoded_v_masked = loc_decoded_v[idxs]
            #loc_decoded_v_masked = loc_decoded_v[torch.stack(idxs,0)]
            
            loc_decoded_v_masked_abs = loc_decoded_v_masked *torch.tensor([w,h,w,h]).float()
            
            mv_conf_prob, mi_conf_prob = conf_prob_v[idxs].max(1)
            
            
            loc_decoded_v = loc_decoded_v[idxs]
            out = {'percent_box': loc_decoded_v,
                   'absolute_box': loc_decoded_v_masked_abs,
                   'confidence': mv_conf_prob,
                   'class': [self.cls_list[i - 1] for i in mi_conf_prob],
                   'width': w,
                   'height': h}
            output_list.append(out)
        return output_list

    def mAP_setup(self, source, target = 'predicted', idx_file='trainval',
                  verbose = True):
        '''
        Setup baseline mAP computing program file format.
        https://github.com/Cartucho/mAP
        
        source is root of VOC dataset.
        Corresponding predicted value will be written into target folder.
        '''
        datasets = DetectionTestDataset(source, self.cls_list, idx_file = idx_file,
                         transform = test_transform)
        dataloader = data.DataLoader(datasets, 
                        batch_size = config.batch_size,
                        shuffle = False,
                        num_workers = config.num_workers,
                        collate_fn = DetectionTestDataset.collate_fn)
        
        for batch_idx,(imgs, meta_list) in enumerate(dataloader):
            det_list = self._detect_batch(imgs, meta_list)
            for detection, meta in zip(det_list, meta_list):
                fname = os.path.split(meta['path'])[1]
                img_idx, _ = os.path.splitext(fname)
                target_path = os.path.join(target, img_idx+'.txt')
                with open(target_path,'w') as f:
                    #for obj in detection:
                    for cls, conf, (left,top,right,bottom) in zip(detection['class'],
                                       detection['confidence'], detection['absolute_box']):
                        #cls = obj['class']
                        #left,top,right,bottom = obj['absolute_box']
                        #conf = obj['confidence']
                        line = '{} {} {} {} {} {}\n'.format(cls, conf, left, top, right, bottom)
                        f.write(line)
                if verbose:
                    print('{} -> {}'.format(meta['path'], target_path))
                    

        
        