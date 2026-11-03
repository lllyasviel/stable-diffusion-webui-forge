# A reimplemented version in public environments by Xiao Fu and Mu Hu

from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from geo_models.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from geo_utils.image_util import resize_max_res,chw2hwc,colorize_depth_maps
from geo_utils.colormap import kitti_colormap
from geo_utils.depth_ensemble import ensemble_depths
from geo_utils.normal_ensemble import ensemble_normals
from geo_utils.batch_size import find_batch_size
import cv2

class DepthNormalPipelineOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        normal_np (`np.ndarray`):
            Predicted normal map, with depth values in the range of [0, 1].
        normal_colored (`PIL.Image.Image`):
            Colorized normal map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    depth_np: np.ndarray
    depth_colored: Image.Image
    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]

class DepthNormalEstimationPipeline(DiffusionPipeline):
    # two hyper-parameters
    latent_scale_factor = 0.18215

    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDIMScheduler,
                 image_encoder:CLIPVisionModelWithProjection,
                 feature_extractor:CLIPImageProcessor,
                 ):
        super().__init__()
            
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.img_embed = None  

    @torch.no_grad()
    def __call__(self,
                 input_image:Image,
                 denoising_steps: int = 10,
                 ensemble_size: int = 10,
                 processing_res: int = 768,
                 match_input_res:bool =True,
                 batch_size:int = 0,
                 domain: str = "indoor",
                 color_map: str="Spectral",
                 show_progress_bar:bool = True,
                 ensemble_kwargs: Dict = None,
                 ) -> DepthNormalPipelineOutput:
        
        # inherit from thea Diffusion Pipeline
        device = self.device
        input_size = input_image.size
        
        # adjust the input resolution.
        if not match_input_res:
            assert (
                processing_res is not None                
            )," Value Error: `resize_output_back` is only valid with "
        
        assert processing_res >=0
        assert denoising_steps >=1
        assert ensemble_size >=1

        # --------------- Image Processing ------------------------
        # Resize image
        if processing_res >0:
            input_image = resize_max_res(
                input_image, max_edge_resolution=processing_res
            )
        
        # Convert the image to RGB, to 1. reomve the alpha channel.
        input_image = input_image.convert("RGB")
        image = np.array(input_image)

        # Normalize RGB Values.
        rgb = np.transpose(image,(2,0,1))
        rgb_norm = rgb / 255.0 * 2.0 - 1.0 # [0, 255] -> [-1, 1]
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)

        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
        
        # ----------------- predicting depth -----------------
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        
        # find the batch size
        if batch_size>0:
            _bs = batch_size
        else:
            _bs = 1

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)
        
        # predicted the depth
        depth_pred_ls = []
        normal_pred_ls = []
        
        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader
        
        for batch in iterable_bar:
            (batched_image, )= batch  # here the image is still around 0-1

            depth_pred_raw, normal_pred_raw = self.single_infer(
                input_rgb=batched_image,
                num_inference_steps=denoising_steps,
                domain=domain,
                show_pbar=show_progress_bar,
            )
            depth_pred_ls.append(depth_pred_raw.detach().clone())
            normal_pred_ls.append(normal_pred_raw.detach().clone())
        
        depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze() #(10,224,768)
        normal_preds = torch.concat(normal_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depths(
                depth_preds, **(ensemble_kwargs or {})
            )
            normal_pred = ensemble_normals(normal_preds)
        else:
            depth_pred = depth_preds
            normal_pred = normal_preds
            pred_uncert = None

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.min(depth_pred)
        max_d = torch.max(depth_pred)
        depth_pred = (depth_pred - min_d) / (max_d - min_d)
        
        # Convert to numpy
        depth_pred = depth_pred.cpu().numpy().astype(np.float32)
        normal_pred = normal_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if match_input_res:
            pred_img = Image.fromarray(depth_pred)
            pred_img = pred_img.resize(input_size)
            depth_pred = np.asarray(pred_img)
            normal_pred = cv2.resize(chw2hwc(normal_pred), input_size, interpolation = cv2.INTER_NEAREST)

        # Clip output range: current size is the original size
        depth_pred = depth_pred.clip(0, 1)
        normal_pred = normal_pred.clip(-1, 1)
    
        # Colorize
        depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap=color_map
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)

        normal_colored = ((normal_pred + 1)/2 * 255).astype(np.uint8)
        normal_colored_img = Image.fromarray(normal_colored)

        self.img_embed = None
        
        return DepthNormalPipelineOutput(
            depth_np = depth_pred,
            depth_colored = depth_colored_img,
            normal_np = normal_pred,
            normal_colored = normal_colored_img,
            uncertainty=pred_uncert,
        )
    
    def __encode_img_embed(self, rgb):
        """
        Encode clip embeddings for img
        """
        clip_image_mean = torch.as_tensor(self.feature_extractor.image_mean)[:,None,None].to(device=self.device, dtype=self.dtype)
        clip_image_std = torch.as_tensor(self.feature_extractor.image_std)[:,None,None].to(device=self.device, dtype=self.dtype)

        img_in_proc = TF.resize((rgb +1)/2, 
            (self.feature_extractor.crop_size['height'], self.feature_extractor.crop_size['width']), 
            interpolation=InterpolationMode.BICUBIC, 
            antialias=True
        )
        # do the normalization in float32 to preserve precision
        img_in_proc = ((img_in_proc.float() - clip_image_mean) / clip_image_std).to(self.dtype)        
        img_embed = self.image_encoder(img_in_proc).image_embeds.unsqueeze(1).to(self.dtype)

        self.img_embed = img_embed

        
    @torch.no_grad()
    def single_infer(self,input_rgb:torch.Tensor,
                     num_inference_steps:int,
                     domain:str,
                     show_pbar:bool,):

        device = input_rgb.device

        # Set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]
        
        # encode image
        rgb_latent = self.encode_RGB(input_rgb)
        
        # Initial geometric maps (Guassian noise)
        geo_latent = torch.randn(rgb_latent.shape, device=device, dtype=self.dtype).repeat(2,1,1,1)
        rgb_latent = rgb_latent.repeat(2,1,1,1)

        # Batched img embedding
        if self.img_embed is None:
            self.__encode_img_embed(input_rgb)
        
        batch_img_embed = self.img_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 1, 768]

        # hybrid switcher 
        geo_class = torch.tensor([[0., 1.], [1, 0]], device=device, dtype=self.dtype)
        geo_embedding = torch.cat([torch.sin(geo_class), torch.cos(geo_class)], dim=-1)
        
        if domain == "indoor":
            domain_class = torch.tensor([[1., 0., 0]], device=device, dtype=self.dtype).repeat(2,1)
        elif domain == "outdoor":
            domain_class = torch.tensor([[0., 1., 0]], device=device, dtype=self.dtype).repeat(2,1)
        elif domain == "object":
            domain_class = torch.tensor([[0., 0., 1]], device=device, dtype=self.dtype).repeat(2,1)
        domain_embedding = torch.cat([torch.sin(domain_class), torch.cos(domain_class)], dim=-1)

        class_embedding = torch.cat((geo_embedding, domain_embedding), dim=-1)

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        
        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, geo_latent], dim=1)

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t.repeat(2), encoder_hidden_states=batch_img_embed, class_labels=class_embedding
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            geo_latent = self.scheduler.step(noise_pred, t, geo_latent).prev_sample

        geo_latent = geo_latent
        torch.cuda.empty_cache()

        depth = self.decode_depth(geo_latent[0][None])
        depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0
        
        normal = self.decode_normal(geo_latent[1][None])
        normal /= (torch.norm(normal, p=2, dim=1, keepdim=True)+1e-5)
        normal *= -1.

        return depth, normal
        
    
    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """

        # encode
        h = self.vae.encoder(rgb_in)

        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.latent_scale_factor
        
        return rgb_latent
    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """

        # scale latent
        depth_latent = depth_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode normal latent into normal map.

        Args:
            normal_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded normal map.
        """

        # scale latent
        normal_latent = normal_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(normal_latent)
        normal = self.vae.decoder(z)
        return normal


