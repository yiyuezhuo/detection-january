# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:59:09 2019

@author: yiyuezhuo
"""

import torch.utils.data as data
import os
import imageio
import xml.etree.ElementTree as ET
import numpy as np


def parseVOC(path):
    # example: classes = ['open', 'close']
    root = ET.parse(path).getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin = bndbox.find('ymin').text
        ymax = bndbox.find('ymax').text
        objects.append((name,int(xmin),int(ymin),int(xmax),int(ymax)))
        
    return objects


class DetectionDataset(data.Dataset):
    '''
    Abstract Pascal VOC like dataset for detection
    '''
    def __init__(self, root, name_list, idx_file='trainval', transform=None):
        self.root = root
        self.name_list = name_list
        self.name2idx = {name:idx for idx,name in enumerate(name_list)}
        self.idx_file = idx_file
        
        idx_file_path = os.path.join(root, 'ImageSets', 'Main', idx_file+'.txt')
        with open(idx_file_path) as f:
            self.idxs = [line.strip() for line in f.readlines()]
        
        self.transform=transform
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, list_idx):
        idx = self.idxs[list_idx]
        img_path = os.path.join(self.root, 'JPEGImages', idx+'.jpg')
        anno_path = os.path.join(self.root, 'Annotations', idx+'.xml')
        
        img = imageio.imread(img_path)
        anno = parseVOC(anno_path)
        
        labels = []
        coords = []
        for obj in anno:
            labels.append(self.name2idx[obj[0]])
            coords.append(obj[1:])
        labels = np.array(labels)
        coords = np.array(coords)
        
        if self.transform is not None:
            img, coords, labels = self.transform(img, coords, labels)
        
        return img, coords, labels
        
if __name__ == '__main__':
    dataset = DetectionDataset(r'E:\agent3\lab\switch', ['open', 'close'])
    for c,d,e in dataset:
        break