# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:25:58 2019

@author: yiyuezhuo
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets


resnet18 = models.resnet18()
pth = torch.load('resnet18-5c106cde.pth')
resnet18.load_state_dict(pth)
torch.save(resnet18.state_dict(),'weights/resnet18.pth') # To see whether the size is as same as prebious one.

class ResNetReduced(models.ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x
    
resnet18.__class__ = ResNetReduced

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('images', transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=True,
    num_workers=1, pin_memory=True)
    
for a,b in train_loader:
    break

print(resnet18(a).shape)