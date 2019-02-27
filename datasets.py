# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:59:09 2019

@author: yiyuezhuo
"""

import torch
import torch.utils.data as data
import os
import imageio
import xml.etree.ElementTree as ET
import numpy as np
from transforms import train_transform
import shutil
#import torch

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
        '''
        According to transform, it may return numpy.array or torch.tensor.
        '''
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
    
    @staticmethod
    def collate_fn(batch):
        '''
        Helper function may be used in DataLoader
        '''
        img_list = []
        coords_list = []
        labels_list = []
        for img, coords, labels in batch:
            img_list.append(img)
            coords_list.append(coords)
            labels_list.append(labels)
        return torch.stack(img_list, 0), coords_list, labels_list
    
    def copy_annotations(self, target, verbose=True):
        for list_idx in range(len(self)):
            idx = self.idxs[list_idx]
            anno_path = os.path.join(self.root, 'Annotations', idx+'.xml')
            fname = os.path.split(anno_path)[-1]
            target_path = os.path.join(target, fname)
            shutil.copy(anno_path, target_path)
            if verbose:
                print('{} -> {}'.format(anno_path, target_path))
            
            

    
class SwitchDatasets(DetectionDataset):
    def __init__(self, root, idx_file='trainval'):
        super().__init__(root, ['open', 'close'], idx_file = idx_file,
             transform = train_transform)
        
class DetectionTestDataset(DetectionDataset):
    '''
    The class suppose VOC format, but will not output annotation, but some
    meta information including path, width and height of images.
    
    The transform used here can only accept image as parameter.
    '''
    def __getitem__(self, list_idx):
        '''
        According to transform, it may return numpy.array or torch.tensor.
        '''
        idx = self.idxs[list_idx]
        img_path = os.path.join(self.root, 'JPEGImages', idx+'.jpg')
        
        img = imageio.imread(img_path)
        meta = {'path': img_path,
                'height': img.shape[0],
                'width': img.shape[1]}
        
        if self.transform is not None:
            img = self.transform(img) 
        
        return img, meta
    
    @staticmethod
    def collate_fn(batch):
        '''
        Helper function may be used in DataLoader
        '''
        img_list = []
        meta_list = []
        for img, meta in batch:
            img_list.append(img)
            meta_list.append(meta)
        return torch.stack(img_list, 0), meta_list
    


    
        
class TestDatasets(data.Dataset):
    '''
    TestDatasets is not VOC format, but a abstraction of a folder saving 
    images. It don't require labels or other meta information.
    '''
    def __init__(self, root):
        self.root = root
        self.fname_list = os.listdir(root)
    def __len__(self):
        return len(self.fname_list)
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fname_list[idx])
        return imageio.imread(path)
        

if __name__ == '__main__':
    dataset = DetectionDataset(r'E:\agent3\lab\switch', ['open', 'close'], 
                               transform= train_transform)
    for c,d,e in dataset:
        break