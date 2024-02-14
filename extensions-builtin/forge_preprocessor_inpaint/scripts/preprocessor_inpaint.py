import os
import cv2
import torch
import numpy as np
import yaml
import einops

from omegaconf import OmegaConf
from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.forge_util import numpy_to_pytorch, resize_image_with_pad
from modules_forge.shared import preprocessor_dir, add_supported_preprocessor
from modules.modelloader import load_file_from_url
from annotator.lama.saicinpainting.training.trainers import load_checkpoint


class PreprocessorInpaint(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'inpaint_global_harmonious'
        self.tags = ['Inpaint']
        self.model_filename_filters = ['inpaint']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.fill_mask_with_one_when_resize_and_fill = True
        self.expand_mask_when_resize_and_fill = True

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        mask = mask.round()
        mixed_cond = cond * (1.0 - mask) - mask
        return mixed_cond, None


class PreprocessorInpaintOnly(PreprocessorInpaint):
    def __init__(self):
        super().__init__()
        self.name = 'inpaint_only'
        self.image = None
        self.mask = None
        self.latent = None

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        mask = mask.round()
        self.image = cond
        self.mask = mask

        vae = process.sd_model.forge_objects.vae
        # This is a powerful VAE with integrated memory management, bf16, and tiled fallback.

        latent_image = vae.encode(self.image.movedim(1, -1))
        latent_image = process.sd_model.forge_objects.unet.model.latent_format.process_in(latent_image)

        B, C, H, W = latent_image.shape

        latent_mask = self.mask
        latent_mask = torch.nn.functional.interpolate(latent_mask, size=(H * 8, W * 8), mode="bilinear").round()
        latent_mask = torch.nn.functional.max_pool2d(latent_mask, (8, 8)).round().to(latent_image)

        unet = process.sd_model.forge_objects.unet.clone()

        def pre_cfg(model, c, uc, x, timestep, model_options):
            noisy_latent = latent_image.to(x) + timestep[:, None, None, None].to(x) * torch.randn_like(latent_image).to(x)
            x = x * latent_mask.to(x) + noisy_latent.to(x) * (1.0 - latent_mask.to(x))
            return model, c, uc, x, timestep, model_options

        def post_cfg(args):
            denoised = args['denoised']
            denoised = denoised * latent_mask.to(denoised) + latent_image.to(denoised) * (1.0 - latent_mask.to(denoised))
            return denoised

        unet.add_sampler_pre_cfg_function(pre_cfg)
        unet.set_model_sampler_post_cfg_function(post_cfg)

        process.sd_model.forge_objects.unet = unet

        self.latent = latent_image

        mixed_cond = cond * (1.0 - mask) - mask

        return mixed_cond, None

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        a1111_batch_result = args[0]
        new_results = []

        for img in a1111_batch_result.images:
            sigma = 7
            mask = self.mask[0, 0].detach().cpu().numpy().astype(np.float32)
            mask = cv2.dilate(mask, np.ones((sigma, sigma), dtype=np.uint8))
            mask = cv2.blur(mask, (sigma, sigma))[None]
            mask = torch.from_numpy(np.ascontiguousarray(mask).copy()).to(img).clip(0, 1)
            raw = self.image[0].to(img).clip(0, 1)
            img = img.clip(0, 1)
            new_results.append(raw * (1.0 - mask) + img * mask)

        a1111_batch_result.images = new_results
        return


class PreprocessorInpaintLama(PreprocessorInpaintOnly):
    def __init__(self):
        super().__init__()
        self.name = 'inpaint_only+lama'

    def load_model(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth"
        model_path = load_file_from_url(remote_model_path, model_dir=preprocessor_dir)
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lama_config.yaml')
        cfg = yaml.safe_load(open(config_path, 'rt'))
        cfg = OmegaConf.create(cfg)
        cfg.training_model.predict_only = True
        cfg.visualizer.kind = 'noop'
        model = load_checkpoint(cfg, os.path.abspath(model_path), strict=False, map_location='cpu')
        self.setup_model_patcher(model)
        return

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, input_mask=None, **kwargs):
        if input_mask is None:
            return input_image

        H, W, C = input_image.shape
        raw_color = input_image.copy()
        raw_mask = input_mask.copy()

        input_image, remove_pad = resize_image_with_pad(input_image, 256)
        input_mask, remove_pad = resize_image_with_pad(input_mask, 256)
        input_mask = input_mask[..., :1]

        self.load_model()

        self.move_all_model_patchers_to_gpu()

        color = np.ascontiguousarray(input_image).astype(np.float32) / 255.0
        mask = np.ascontiguousarray(input_mask).astype(np.float32) / 255.0
        with torch.no_grad():
            color = self.send_tensor_to_model_device(torch.from_numpy(color))
            mask = self.send_tensor_to_model_device(torch.from_numpy(mask))
            mask = (mask > 0.5).float()
            color = color * (1 - mask)
            image_feed = torch.cat([color, mask], dim=2)
            image_feed = einops.rearrange(image_feed, 'h w c -> 1 c h w')
            prd_color = self.model_patcher.model(image_feed)[0]
            prd_color = einops.rearrange(prd_color, 'c h w -> h w c')
            prd_color = prd_color * mask + color * (1 - mask)
            prd_color *= 255.0
            prd_color = prd_color.detach().cpu().numpy().clip(0, 255).astype(np.uint8)

        prd_color = remove_pad(prd_color)
        prd_color = cv2.resize(prd_color, (W, H))

        alpha = raw_mask.astype(np.float32) / 255.0
        fin_color = prd_color.astype(np.float32) * alpha + raw_color.astype(np.float32) * (1 - alpha)
        fin_color = fin_color.clip(0, 255).astype(np.uint8)

        return fin_color

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        cond, mask = super().process_before_every_sampling(process, cond, mask, *args, **kwargs)
        sigma_max = process.sd_model.forge_objects.unet.model.model_sampling.sigma_max
        original_noise = kwargs['noise']
        process.modified_noise = original_noise + self.latent.to(original_noise) / sigma_max.to(original_noise)
        return cond, mask


add_supported_preprocessor(PreprocessorInpaint())

add_supported_preprocessor(PreprocessorInpaintOnly())

add_supported_preprocessor(PreprocessorInpaintLama())
