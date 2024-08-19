# A reimplemented version in public environments by Xiao Fu and Mu Hu

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

from dataloader.mix_loader import MixDataset
from torch.utils.data import DataLoader
from dataloader import transforms
import os


# Get Dataset Here
def prepare_dataset(data_dir=None,
                    batch_size=1,
                    test_batch=1,
                    datathread=4,
                    logger=None):

    # set the config parameters
    dataset_config_dict = dict()
    
    train_dataset = MixDataset(data_dir=data_dir)

    img_height, img_width = train_dataset.get_img_size()

    datathread = datathread
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    
    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)
    
    num_batches_per_epoch = len(train_loader)
    
    dataset_config_dict['num_batches_per_epoch'] = num_batches_per_epoch
    dataset_config_dict['img_size'] = (img_height,img_width)
    
    return train_loader, dataset_config_dict

def depth_scale_shift_normalization(depth):

    bsz = depth.shape[0]

    depth_ = depth[:,0,:,:].reshape(bsz,-1).cpu().numpy()
    min_value = torch.from_numpy(np.percentile(a=depth_,q=2,axis=1)).to(depth)[...,None,None,None]
    max_value = torch.from_numpy(np.percentile(a=depth_,q=98,axis=1)).to(depth)[...,None,None,None]

    normalized_depth = ((depth - min_value)/(max_value-min_value+1e-5) - 0.5) * 2
    normalized_depth = torch.clip(normalized_depth, -1., 1.)

    return normalized_depth



def resize_max_res_tensor(input_tensor, mode, recom_resolution=768):
    assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]
    downscale_factor = min(recom_resolution/original_H, recom_resolution/original_W)
    
    if mode == 'normal':
        resized_input_tensor = F.interpolate(input_tensor,
                                            scale_factor=downscale_factor,
                                            mode='nearest')
    else:
        resized_input_tensor = F.interpolate(input_tensor,
                                            scale_factor=downscale_factor,
                                            mode='bilinear',
                                            align_corners=False)

    if mode == 'depth':
        return resized_input_tensor / downscale_factor
    else:
        return resized_input_tensor
