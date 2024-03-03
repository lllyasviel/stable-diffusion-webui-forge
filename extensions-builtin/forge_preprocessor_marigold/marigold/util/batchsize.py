# Author: Bingxin Ke
# Last modified: 2023-12-11

import torch
import math


# Search table for suggested max. inference batch size
bs_search_table = [
    # tested on A100-PCIE-80GB
    {"res": 768, "total_vram": 79, "bs": 35},
    {"res": 1024, "total_vram": 79, "bs": 20},
    # tested on A100-PCIE-40GB
    {"res": 768, "total_vram": 39, "bs": 15},
    {"res": 1024, "total_vram": 39, "bs": 8},
    # tested on RTX3090, RTX4090
    {"res": 512, "total_vram": 23, "bs": 20},   
    {"res": 768, "total_vram": 23, "bs": 7},
    {"res": 1024, "total_vram": 23, "bs": 3},
    # tested on GTX1080Ti
    {"res": 512, "total_vram": 10, "bs": 5},
    {"res": 768, "total_vram": 10, "bs": 2},
]



def find_batch_size(n_repeat, input_res):
    total_vram = torch.cuda.mem_get_info()[1] / 1024.0**3
    
    for settings in sorted(bs_search_table, key=lambda k: (k['res'], -k['total_vram'])):
        if input_res <= settings['res'] and total_vram >= settings['total_vram']:
            bs = settings['bs']
            if bs > n_repeat:
                bs = n_repeat
            elif bs > math.ceil(n_repeat / 2) and bs < n_repeat:
                bs = math.ceil(n_repeat / 2)
            return bs
    return 1