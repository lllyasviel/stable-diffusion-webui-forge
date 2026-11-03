# Author: Bingxin Ke
# Last modified: 2023-12-11

import logging
from typing import Dict

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    DEISMultistepScheduler,
    SchedulerMixin,
    UNet2DConditionModel,
)
from torch import nn
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .rgb_encoder import RGBEncoder
from .stacked_depth_AE import StackedDepthAE


class MarigoldPipeline(nn.Module):
    """
    Marigold monocular depth estimator.
    """

    def __init__(
        self,
        unet_pretrained_path: Dict,  # {path: xxx, subfolder: xxx}
        rgb_encoder_pretrained_path: Dict,
        depht_ae_pretrained_path: Dict,
        noise_scheduler_pretrained_path: Dict,
        tokenizer_pretrained_path: Dict,
        text_encoder_pretrained_path: Dict,
        empty_text_embed=None,
        trainable_unet=False,
        rgb_latent_scale_factor=0.18215,
        depth_latent_scale_factor=0.18215,
        noise_scheduler_type=None,
        enable_gradient_checkpointing=False,
        enable_xformers=True,
    ) -> None:
        super().__init__()

        self.rgb_latent_scale_factor = rgb_latent_scale_factor
        self.depth_latent_scale_factor = depth_latent_scale_factor
        self.device = "cpu"

        # ******* Initialize modules *******
        # Trainable modules
        self.trainable_module_dic: Dict[str, nn.Module] = {}
        self.trainable_unet = trainable_unet

        # Denoising UNet
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            unet_pretrained_path["path"], subfolder=unet_pretrained_path["subfolder"]
        )
        logging.info(f"pretrained UNet loaded from: {unet_pretrained_path}")
        if 8 != self.unet.config["in_channels"]:
            self._replace_unet_conv_in()
            logging.warning("Unet conv_in layer is replaced")
        if enable_xformers:
            self.unet.enable_xformers_memory_efficient_attention()
        else:
            self.unet.disable_xformers_memory_efficient_attention()

        # Image encoder
        self.rgb_encoder = RGBEncoder(
            pretrained_path=rgb_encoder_pretrained_path["path"],
            subfolder=rgb_encoder_pretrained_path["subfolder"],
        )
        logging.info(
            f"pretrained RGBEncoder loaded from: {rgb_encoder_pretrained_path}"
        )
        self.rgb_encoder.requires_grad_(False)

        # Depth encoder-decoder
        self.depth_ae = StackedDepthAE(
            pretrained_path=depht_ae_pretrained_path["path"],
            subfolder=depht_ae_pretrained_path["subfolder"],
        )
        logging.info(
            f"pretrained Depth Autoencoder loaded from: {rgb_encoder_pretrained_path}"
        )

        # Trainability
        # unet
        if self.trainable_unet:
            self.unet.requires_grad_(True)
            self.trainable_module_dic["unet"] = self.unet
            logging.debug(f"UNet is set to trainable")
        else:
            self.unet.requires_grad_(False)
            logging.debug(f"UNet is set to frozen")

        # Gradient checkpointing
        if enable_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            self.depth_ae.vae.enable_gradient_checkpointing()

        # Noise scheduler
        if "DDPMScheduler" == noise_scheduler_type:
            self.noise_scheduler: SchedulerMixin = DDPMScheduler.from_pretrained(
                noise_scheduler_pretrained_path["path"],
                subfolder=noise_scheduler_pretrained_path["subfolder"],
            )
        elif "DDIMScheduler" == noise_scheduler_type:
            self.noise_scheduler: SchedulerMixin = DDIMScheduler.from_pretrained(
                noise_scheduler_pretrained_path["path"],
                subfolder=noise_scheduler_pretrained_path["subfolder"],
            )
        elif "PNDMScheduler" == noise_scheduler_type:
            self.noise_scheduler: SchedulerMixin = PNDMScheduler.from_pretrained(
                noise_scheduler_pretrained_path["path"],
                subfolder=noise_scheduler_pretrained_path["subfolder"],
            )
        elif "DEISMultistepScheduler" == noise_scheduler_type:
            self.noise_scheduler: SchedulerMixin = DEISMultistepScheduler.from_pretrained(
                noise_scheduler_pretrained_path["path"],
                subfolder=noise_scheduler_pretrained_path["subfolder"],
            )
        else:
            raise NotImplementedError

        # Text embed for empty prompt (always in CPU)
        if empty_text_embed is None:
            tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_pretrained_path["path"],
                subfolder=tokenizer_pretrained_path["subfolder"],
            )
            text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
                text_encoder_pretrained_path["path"],
                subfolder=text_encoder_pretrained_path["subfolder"],
            )
            with torch.no_grad():
                self.empty_text_embed = self._encode_text(
                    "", tokenizer, text_encoder
                ).detach()#.to(dtype=precision)  # [1, 2, 1024]
        else:
            self.empty_text_embed = empty_text_embed

    def from_pretrained(pretrained_path, **kwargs):
        return __class__(
            unet_pretrained_path={"path": pretrained_path, "subfolder": "unet"},
            rgb_encoder_pretrained_path={"path": pretrained_path, "subfolder": "vae"},
            depht_ae_pretrained_path={"path": pretrained_path, "subfolder": "vae"},
            noise_scheduler_pretrained_path={
                "path": pretrained_path,
                "subfolder": "scheduler",
            },
            tokenizer_pretrained_path={
                "path": pretrained_path,
                "subfolder": "tokenizer",
            },
            text_encoder_pretrained_path={
                "path": pretrained_path,
                "subfolder": "text_encoder",
            },
            **kwargs,
        )

    def _replace_unet_conv_in(self):
        # Replace the first layer to accept 8 in_channels. Only applied when loading pretrained SD U-Net
        _weight = self.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.5
        _bias *= 0.5
        # new conv_in channel
        _n_convin_out_channel = self.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.unet.conv_in = _new_conv_in
        # replace config
        self.unet.config["in_channels"] = 8
        return

    def to(self, device):
        self.rgb_encoder.to(device)
        self.depth_ae.to(device)
        self.unet.to(device)
        self.empty_text_embed = self.empty_text_embed.to(device)
        self.device = device
        return self

    def forward(
        self,
        rgb_in,
        num_inference_steps: int = 50,
        num_output_inter_results: int = 0,
        show_pbar=False,
        init_depth_latent=None,
        return_depth_latent=False,
    ):
        device = rgb_in.device
        precision = self.unet.dtype    
        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial depth map (noise)
        if init_depth_latent is not None:
            init_depth_latent = init_depth_latent.to(dtype=precision)
            assert (
                init_depth_latent.shape == rgb_latent.shape
            ), "initial depth latent should be the size of [B, 4, H/8, W/8]"
            depth_latent = init_depth_latent
            depth_latent = torch.randn(rgb_latent.shape, device=device, dtype=precision)
        else:
            depth_latent = torch.randn(rgb_latent.shape, device=device)  # [B, 4, h, w]

        # Expand text embeding for batch
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device=device, dtype=precision)  # [B, 2, 1024]

        # Export intermediate denoising steps
        if num_output_inter_results > 0:
            depth_latent_ls = []
            inter_steps = []
            _idx = (
                -1
                * (
                    np.arange(0, num_output_inter_results)
                    * num_inference_steps
                    / num_output_inter_results
                )
                .round()
                .astype(int)
                - 1
            )
            steps_to_output = timesteps[_idx]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(enumerate(timesteps), total=len(timesteps), leave=False, desc="denoising")
        else:
            iterable = enumerate(timesteps)
        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )  # this order is important
            unet_input = unet_input.to(dtype=precision)
            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]
            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.noise_scheduler.step(
                noise_pred, t, depth_latent
            ).prev_sample.to(dtype=precision)
            

            if num_output_inter_results > 0 and t in steps_to_output:
                depth_latent_ls.append(depth_latent.detach().clone())
                #depth_latent_ls = depth_latent_ls.to(dtype=precision)
                inter_steps.append(t - 1)

        # Decode depth latent
        if num_output_inter_results > 0:
            assert 0 in inter_steps
            depth = [self.decode_depth(lat) for lat in depth_latent_ls]
            if return_depth_latent:
                return depth, inter_steps, depth_latent_ls
            else:
                return depth, inter_steps
        else:
            depth = self.decode_depth(depth_latent)
            if return_depth_latent:
                return depth, depth_latent
            else:
                return depth

    def encode_rgb(self, rgb_in):
        rgb_latent = self.rgb_encoder(rgb_in)  # [B, 4, h, w]
        rgb_latent = rgb_latent * self.rgb_latent_scale_factor
        return rgb_latent 

    def encode_depth(self, depth_in):
        depth_latent = self.depth_ae.encode(depth_in)
        depth_latent = depth_latent * self.depth_latent_scale_factor
        return depth_latent

    def decode_depth(self, depth_latent):
        #depth_latent = depth_latent.to(dtype=torch.float16)
        depth_latent = depth_latent / self.depth_latent_scale_factor
        depth = self.depth_ae.decode(depth_latent)  # [B, 1, H, W]
        return depth 

    @staticmethod
    def _encode_text(prompt, tokenizer, text_encoder):
        text_inputs = tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(text_encoder.device)
        text_embed = text_encoder(text_input_ids)[0]
        return text_embed
