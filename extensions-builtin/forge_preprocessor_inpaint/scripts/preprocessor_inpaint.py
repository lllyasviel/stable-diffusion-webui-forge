import torch

from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.forge_util import numpy_to_pytorch


class PreprocessorInpaint(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'inpaint_global_harmonious'
        self.tags = ['Inpaint']
        self.model_filename_filters = ['inpaint']
        self.slider_resolution = PreprocessorParameter(visible=False)


class PreprocessorInpaintOnly(PreprocessorInpaint):
    def __init__(self):
        super().__init__()
        self.name = 'inpaint_only'
        self.image = None
        self.mask = None

    def process_before_every_sampling(self, process, cond, *args, **kwargs):
        self.image = kwargs['cond_before_inpaint_fix'][:, 0:3]
        self.mask = kwargs['cond_before_inpaint_fix'][:, 3:]

        vae = process.sd_model.forge_objects.vae
        # This is a powerful VAE with integrated memory management, bf16, and tiled fallback.

        latent_image = vae.encode(self.image.movedim(1, -1))
        latent_image = process.sd_model.forge_objects.unet.model.latent_format.process_in(latent_image)

        B, C, H, W = latent_image.shape

        latent_mask = self.mask
        latent_mask = torch.nn.functional.interpolate(latent_mask, size=(H * 8, W * 8), mode="bilinear").round()
        latent_mask = torch.nn.functional.max_pool2d(latent_mask, (8, 8)).round().to(latent_image)

        unet = process.sd_model.forge_objects.unet.clone()

        def post_cfg(args):
            denoised = args['denoised']
            denoised = denoised * latent_mask.to(denoised) + latent_image.to(denoised) * (1.0 - latent_mask.to(denoised))
            return denoised

        unet.set_model_sampler_post_cfg_function(post_cfg)

        process.sd_model.forge_objects.unet = unet
        return

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        return


class PreprocessorInpaintLama(PreprocessorInpaintOnly):
    def __init__(self):
        super().__init__()
        self.name = 'inpaint_only+lama'


add_supported_preprocessor(PreprocessorInpaint())

add_supported_preprocessor(PreprocessorInpaintOnly())

add_supported_preprocessor(PreprocessorInpaintLama())
