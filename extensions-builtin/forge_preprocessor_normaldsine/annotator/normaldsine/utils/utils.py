""" utils
"""
import os
import torch
import numpy as np


def load_checkpoint(fpath, model):
    print('loading checkpoint... {}'.format(fpath))

    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    print('loading checkpoint... / done')
    return model


def compute_normal_error(pred_norm, gt_norm):
    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi
    pred_error = pred_error.unsqueeze(1)    # (B, 1, H, W)
    return pred_error


def compute_normal_metrics(total_normal_errors):
    total_normal_errors = total_normal_errors.detach().cpu().numpy()
    num_pixels = total_normal_errors.shape[0]

    metrics = {
        'mean': np.average(total_normal_errors),
        'median': np.median(total_normal_errors),
        'rmse': np.sqrt(np.sum(total_normal_errors * total_normal_errors) / num_pixels),
        'a1': 100.0 * (np.sum(total_normal_errors < 5) / num_pixels),
        'a2': 100.0 * (np.sum(total_normal_errors < 7.5) / num_pixels),
        'a3': 100.0 * (np.sum(total_normal_errors < 11.25) / num_pixels),
        'a4': 100.0 * (np.sum(total_normal_errors < 22.5) / num_pixels),
        'a5': 100.0 * (np.sum(total_normal_errors < 30) / num_pixels)
    }

    return metrics


def pad_input(orig_H, orig_W):
    if orig_W % 32 == 0:
        l = 0
        r = 0
    else:
        new_W = 32 * ((orig_W // 32) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 32 == 0:
        t = 0
        b = 0
    else:
        new_H = 32 * ((orig_H // 32) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b


def get_intrins_from_fov(new_fov, H, W, device):
    # NOTE: top-left pixel should be (0,0)
    if W >= H:
        new_fu = (W / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
        new_fv = (W / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    else:
        new_fu = (H / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
        new_fv = (H / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))

    new_cu = (W / 2.0) - 0.5
    new_cv = (H / 2.0) - 0.5

    new_intrins = torch.tensor([
        [new_fu,    0,          new_cu  ],
        [0,         new_fv,     new_cv  ],
        [0,         0,          1       ]
    ], dtype=torch.float32, device=device)

    return new_intrins


def get_intrins_from_txt(intrins_path, device):
    # NOTE: top-left pixel should be (0,0)
    with open(intrins_path, 'r') as f:
        intrins_ = f.readlines()[0].split()[0].split(',')
        intrins_ = [float(i) for i in intrins_]
        fx, fy, cx, cy = intrins_

    intrins = torch.tensor([
        [fx, 0,cx],
        [ 0,fy,cy],
        [ 0, 0, 1]
    ], dtype=torch.float32, device=device)

    return intrins