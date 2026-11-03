# Author: Bingxin Ke
# Last modified: 2023-12-05

import torch
import torch.nn as nn
import logging
from diffusers import AutoencoderKL


class RGBEncoder(nn.Module):
    """
    The encoder of pretrained Stable Diffusion VAE
    """
    
    def __init__(self, pretrained_path, subfolder=None) -> None:
        super().__init__()
        
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(pretrained_path, subfolder=subfolder)
        logging.info(f"pretrained AutoencoderKL loaded from: {pretrained_path}")
        
        self.rgb_encoder = nn.Sequential(
            vae.encoder,
            vae.quant_conv,
        )
    
    def to(self, *args, **kwargs):
        self.rgb_encoder.to(*args, **kwargs)    
    
    def forward(self, rgb_in):
        return self.encode(rgb_in)
    
    def encode(self, rgb_in):
        moments = self.rgb_encoder(rgb_in) # [B, 8, H/8, W/8]
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean
        return rgb_latent