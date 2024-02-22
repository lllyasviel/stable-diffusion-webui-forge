# Author: Bingxin Ke
# Last modified: 2023-12-05

import torch
import torch.nn as nn
import logging
from diffusers import AutoencoderKL


class StackedDepthAE(nn.Module):
    """
    Tailored pretrained image VAE for depth map.
        Encode: Depth images are repeated into 3 channels.
        Decode: The average of 3 chennels are taken as output.
    """

    def __init__(self, pretrained_path, subfolder=None) -> None:
        super().__init__()

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(pretrained_path, subfolder=subfolder)
        logging.info(f"pretrained AutoencoderKL loaded from: {pretrained_path}")

    def forward(self, depth_in):
        depth_latent = self.encode(depth_in)
        depth_out = self.decode(depth_latent)
        return depth_out

    def to(self, *args, **kwargs):
        self.vae.to(*args, **kwargs)

    @staticmethod
    def _stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def encode(self, depth_in):
        stacked = self._stack_depth_images(depth_in)
        h = self.vae.encoder(stacked)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        depth_latent = mean
        return depth_latent

    def decode(self, depth_latent):
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean