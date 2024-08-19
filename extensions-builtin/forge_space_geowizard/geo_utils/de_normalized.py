# A reimplemented version in public environments by Xiao Fu and Mu Hu

import numpy as np
from scipy.optimize import least_squares
import torch

def align_scale_shift(pred, target, clip_max):
    mask = (target > 0) & (target < clip_max)
    if mask.sum() > 10:
        target_mask = target[mask]
        pred_mask = pred[mask]
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        return scale, shift
    else:
        return 1, 0

def align_scale(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    if torch.sum(mask) > 10:
        scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
    else:
        scale = 1
    pred_scale = pred * scale
    return pred_scale, scale

def align_shift(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    if torch.sum(mask) > 10:
        shift = torch.median(target[mask]) - (torch.median(pred[mask]) + 1e-8)
    else:
        shift = 0
    pred_shift = pred + shift
    return pred_shift, shift