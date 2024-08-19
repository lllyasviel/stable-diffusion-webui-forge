# A reimplemented version in public environments by Xiao Fu and Mu Hu

import numpy as np
import torch

def ensemble_normals(input_images:torch.Tensor):
    normal_preds = input_images

    bsz, d, h, w = normal_preds.shape
    normal_preds = normal_preds / (torch.norm(normal_preds, p=2, dim=1).unsqueeze(1)+1e-5)

    phi = torch.atan2(normal_preds[:,1,:,:], normal_preds[:,0,:,:]).mean(dim=0)
    theta = torch.atan2(torch.norm(normal_preds[:,:2,:,:], p=2, dim=1), normal_preds[:,2,:,:]).mean(dim=0)
    normal_pred = torch.zeros((d,h,w)).to(normal_preds)
    normal_pred[0,:,:] = torch.sin(theta) * torch.cos(phi)
    normal_pred[1,:,:] = torch.sin(theta) * torch.sin(phi)
    normal_pred[2,:,:] = torch.cos(theta) 

    angle_error = torch.acos(torch.cosine_similarity(normal_pred[None], normal_preds, dim=1))
    normal_idx = torch.argmin(angle_error.reshape(bsz,-1).sum(-1))

    return normal_preds[normal_idx]