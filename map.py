# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 08:39:30 2019

@author: yiyuezhuo

Calculating mAP

baseline:
    https://github.com/Cartucho/mAP
    The result will be compared with baseline.
    

"""

from utils import iou
import os
import glob
import torch
import numpy as np

def collect(ground_truth_pattern, predicted_pattern):
    '''
    Parse https://github.com/Cartucho/mAP format.
    '''
    ground_truth = {}
    predicted = {}
    for path in glob.glob(ground_truth_pattern):
        fname = os.path.split(path)[-1]
        #g_path = os.path.join(ground_truth_root, fname)
        with open(path) as f:
            g_list = []
            for line in f.readlines():
                cls, left, top, right, bottom = line.split(' ')
                coord = [float(c.strip()) for c in [left,top,right,bottom]]
                g_list.append(dict(cls=cls, coord=coord))
            ground_truth[fname] = g_list
        #p_path = os.path.join(predicted_root, fname)
    for path in glob.glob(predicted_pattern):
        fname = os.path.split(path)[-1]
        with open(path) as f:
            p_list = []
            for line in f.readlines():
                cls, conf, left, top, right, bottom = line.split(' ')
                coord = [float(c.strip()) for c in [conf, left,top,right,bottom]]
                conf = coord[0]
                coord = coord[1:]
                p_list.append(dict(cls=cls, conf=conf, coord=coord))
            predicted[fname] = p_list
    return ground_truth,predicted

def mAP(ground_truth, predicted, threshold = 0.5, classes_list = None):
    '''
    grount_truth = {'file_name.txt': {'cls':str,'coord':(left,top,right,bottom)},...}
    predicted = {'file_name.txt': {'cls':str, 'conf':float, 'coord':(...), fname:str}}
    
    The calling will apply side effect on ground_truth and predicted dict.
    '''
    # setup
        
    predicted_list = []
    for fname, p_list in predicted.items():
        for p in p_list:
            p['fname'] = fname
        predicted_list.extend(p_list)
            
    if classes_list is None:
        classes_list = list(set([p['cls'] for p in predicted_list]))

    total_ground_truth_dict = {c:0 for c in classes_list}
    coords_map = {}
    for fname, g_list in ground_truth.items():
        coord_list = []
        for g in g_list:
            g['matched'] = False
            coord_list.append(g['coord'])
            total_ground_truth_dict[g['cls']] += 1
        coords_map[fname] = torch.tensor(coord_list)
    
            
    # calculating
    predicted_list.sort(key=lambda p:p['conf'], reverse=True)
    match_dict_list = {c:[] for c in classes_list}
    
    for p in predicted_list:
        gboxes = coords_map[p['fname']]
        pboxes = torch.tensor(p['coord']).unsqueeze(0)
        overlap = iou(pboxes, gboxes)[0]
        overlap_sorted, overlap_idx = overlap.sort(descending=True)
        
        matched=False
        for op,idx in zip(overlap_sorted, overlap_idx):
            if op < threshold:
                break
            
            gt = ground_truth[p['fname']][idx]
            if gt['cls'] == p['cls'] and not gt['matched']:
                gt['matched'] = True
                matched = True
                break
        
        match_dict_list[p['cls']].append(matched)
    
    res = {'cls':{}}
    for cls in classes_list:
        # Well, I need np.maximum.accumulate, but I can't find a correspondense on pytorch
        match_array = np.array(match_dict_list[cls]).astype(float)
        cumsum = np.cumsum(match_array)
        accurary_curve = cumsum/(np.arange(match_array.shape[0])+1)
        recall_curve = cumsum/total_ground_truth_dict[cls]
        accuracy_curve_maxed = np.maximum.accumulate(accurary_curve[::-1])[::-1]
        accuracy_inter = (accuracy_curve_maxed[1:] + accuracy_curve_maxed[:-1])/2
        accuracy_range = recall_curve[1:] - recall_curve[:-1]
        acc = np.sum(accuracy_inter * accuracy_range)
        
        res['cls'][cls] = {}
        res['cls'][cls]['accuracy'] = acc
        res['cls'][cls]['accurary_curve'] = accurary_curve
        res['cls'][cls]['recall_curve'] = recall_curve
    
    res['mAP'] = np.mean([cls['accuracy'] for cls in res['cls'].values()])
    
    return res


if __name__ == '__main__':
    ground_truth_pattern = r'E:\agent4\mAP\ground-truth\*.txt'
    predicted_pattern = r'E:\agent4\mAP\predicted\*.txt'
    ground_truth,predicted = collect(ground_truth_pattern, predicted_pattern)
    
    '''
    # test
    first_name = next(iter(ground_truth))
    g = ground_truth[first_name]
    p = predicted[first_name]
    '''
    
    res25 = mAP(ground_truth, predicted, classes_list = ['open', 'close'],
                threshold=0.25)
    res50 = mAP(ground_truth, predicted, classes_list = ['open', 'close'])
    import matplotlib.pyplot as plt
    
    print('mAP = {}'.format(res50['mAP']))
    print('mAP = {} threshold=0.25'.format(res25['mAP']))