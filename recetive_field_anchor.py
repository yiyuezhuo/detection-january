# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:58:28 2019

@author: yiyuezhuo
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from networks import ResNet18Reduced



resnet_features = ResNet18Reduced()
resnet_features.load_state_dict(torch.load('weights/resnet18reduced.pth'))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('images', transforms.Compose([
        transforms.RandomSizedCrop(400),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=True,
    num_workers=1, pin_memory=True)
    
for a,b in train_loader:
    break

resnet_features.eval()

mat = np.empty([7,7,4])
for i in range(7):
    for j in range(7):
        
        ag = a.clone().detach().requires_grad_(True)
        x2,x3 = resnet_features(ag)
        torch.abs(x3).sum(0).sum(0)[i,j].backward()
        #x3.sum(0).sum(0)[i,j].backward()
        g = torch.abs(ag.grad).sum(0).sum(0).detach().numpy()
        
        gg = np.where(np.any(g,axis=0)!=0)[0]
        mat[i,j,0] = gg.min()
        mat[i,j,1] = gg.max()
        
        gg = np.where(np.any(g,axis=1)!=0)[0]
        mat[i,j,2] = gg.min()
        mat[i,j,3] = gg.max()
        break
    break
