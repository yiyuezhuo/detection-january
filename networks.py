# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:09:23 2019

@author: yiyuezhuo
"""

import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def weight_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data)
        module.bias.data.zero_()

class ResNet18Reduced(models.ResNet):
    # models.ResNet is also a nn.Module.
    def __init__(self):
        super(ResNet18Reduced, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2])
        del self.avgpool 
        del self.fc 
        del self.layer4
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = x2 = self.layer2(x)
        x = x3 = self.layer3(x)
        return x2,x3
        #x = x4 = self.layer4(x)
        #return x2,x3,x4
        
class ExportLayer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.loc = nn.Conv2d(in_channels, 4*4, 
                             kernel_size=3,stride=1,padding=1)
        self.conf = nn.Conv2d(in_channels, num_classes*4,
                              kernel_size=3,stride=1,padding=1)
    def forward(self, x):
        return self.loc(x), self.conf(x)
    
class ExtraFeature(nn.Module):
    def __init__(self, in_channels):
        #https://discuss.pytorch.org/t/attributeerror-cannot-assign-module-before-module-init-call-even-if-initialized/33861
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 
                             kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 
                             kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 
                             kernel_size=3,stride=2,padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
          
class JanuaryNet(nn.Module):
    def __init__(self, features, num_classes):
        super().__init__()
        self.features = features # batch_size,
        self.extra = ExtraFeature(256)
        
        self.exports = nn.ModuleList([ExportLayer(128, num_classes),
                                      ExportLayer(256, num_classes),
                                      ExportLayer(256, num_classes)])
    
        self.priors_center_offset, self.priors_point_form = self.generate_prior_boxes()
    
    def init(self, init_fn):
        self.extra.apply(init_fn)
        self.exports.apply(init_fn)
        
    def forward(self, x):
        # For bx3x400x400 input, 
        #x2,x3 = self.features(x) # (b,3,400,400) -> (b,128,50,50), (b,256,25,25)
        x2,x3 = self.features(x)
        x4 = self.extra(x3)
        
        # For 400x400 input
        loc1,conf1 = self.exports[0](x2) # loc: (b, 4*4, 50,50),conf: (b, class*4, 50,50)
        loc2,conf2 = self.exports[1](x3) # loc: (b, 4*4, 25,25),conf: (b, class*4, 25,25)
        loc3,conf3 = self.exports[2](x4) # loc: (b, 4*4, 13,13),conf: (b, class*4, 13,13)
        
        batch_size = loc1.shape[0]
        num_class = conf1.shape[1] // 4 # 4 boxes
        
        locs = []
        for loc in [loc1,loc2,loc3]:
            locs.append(loc.permute(0,2,3,1).contiguous().view(batch_size, -1, 4))
        locs_t = torch.cat(locs, dim=1)
        
        confs = []
        for conf in [conf1, conf2, conf3]:
            confs.append(conf.permute(0,2,3,1).contiguous().view(batch_size,-1, num_class))
        confs_t = torch.cat(confs, dim=1)
            
        return locs_t,confs_t
    
    def generate_prior_boxes(self, ratio=0.75):
        '''
        Generate prior boxes(default boxes) using parameters 
            fk as 50,25,13, sk as 0.15,0.4,0.75.
        
        Return:
            center_offset: [(centerx,centery,width,height),...]
            point_form: [(xmin,ymin,xmax,ymax),...]
        '''
        boxes_list = []
        for fk,sk in zip([50,25,13], [0.15,0.4,0.75]):
            x = (torch.arange(fk).type(torch.float) + 0.5) /fk
            y = (torch.arange(fk).type(torch.float) + 0.5) /fk
            x = x.view(1,fk).unsqueeze(2).expand(fk,fk,4) # (y,x,num_box)
            y = y.view(fk,1).unsqueeze(2).expand(fk,fk,4)
            
            w = torch.tensor([sk, sk*ratio, sk, sk*ratio])
            h = torch.tensor([sk, sk, sk*ratio, sk*ratio])
            w = w.unsqueeze(0).unsqueeze(0).expand(fk,fk,4)
            h = h.unsqueeze(0).unsqueeze(0).expand(fk,fk,4)
            
            priors = torch.stack([x,y,w,h])
            priors = priors.permute(1,2,3,0).contiguous().view(fk*fk*4,4)
            boxes_list.append(priors)
        center_offset = torch.cat(boxes_list, 0)
        
        point_form = torch.empty_like(center_offset)
        point_form[:,0] = center_offset[:,0] - center_offset[:,2]/2
        point_form[:,1] = center_offset[:,1] - center_offset[:,3]/2
        point_form[:,2] = center_offset[:,0] + center_offset[:,2]/2
        point_form[:,3] = center_offset[:,1] + center_offset[:,3]/2
        
        return center_offset,point_form


if __name__ == '__main__':
    # Placing above line is required to solve the issue:
    # https://github.com/pytorch/pytorch/issues/5858
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
    
    #res_res = resnet_features(a)
    
    net = JanuaryNet(resnet_features, 3)
    net.init(weight_init)
    
    loc,conf = net(a)
    