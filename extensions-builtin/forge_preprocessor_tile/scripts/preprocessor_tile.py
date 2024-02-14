import torch

from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor


def blur(x, k):
    y = torch.nn.functional.pad(x, (k, k, k, k), mode='replicate')
    y = torch.nn.functional.avg_pool2d(y, (k * 2 + 1, k * 2 + 1), stride=(1, 1))
    return y


class PreprocessorTile(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'tile_resample'
        self.tags = ['Tile']
        self.model_filename_filters = ['tile']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.latent = None

    def register_latent(self, process, cond):
        vae = process.sd_model.forge_objects.vae
        # This is a powerful VAE with integrated memory management, bf16, and tiled fallback.

        latent_image = vae.encode(cond.movedim(1, -1))
        latent_image = process.sd_model.forge_objects.unet.model.latent_format.process_in(latent_image)
        self.latent = latent_image
        return self.latent


class PreprocessorTileColorFix(PreprocessorTile):
    def __init__(self):
        super().__init__()
        self.name = 'tile_colorfix'
        self.slider_1 = PreprocessorParameter(label='Variation', value=8.0, minimum=3.0, maximum=32.0, step=1.0, visible=True)
        self.variation = 8
        self.sharpness = None

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        self.variation = int(kwargs['unit'].threshold_a)

        latent = self.register_latent(process, cond)

        unet = process.sd_model.forge_objects.unet.clone()
        sigma_data = process.sd_model.forge_objects.unet.model.model_sampling.sigma_data

        if getattr(process, 'is_hr_pass', False):
            k = int(self.variation * 2)
        else:
            k = int(self.variation)

        def block_proc(h, flag, transformer_options):
            location, block_id = transformer_options['block']
            cond_mark = transformer_options['cond_mark'][:, None, None, None]  # cond is 0

            if location == 'input' and block_id == 0 and flag == 'before':
                sigma = transformer_options['sigmas'].to(h)
                self.x_input = h[:, :4]  # Inpaint fix
                self.x_input_sigma_space = self.x_input * (sigma ** 2 + sigma_data ** 2) ** 0.5

            if location == 'last' and block_id == 0 and flag == 'after':
                sigma = transformer_options['sigmas'].to(h)
                eps_estimation = h[:, :4]
                denoised = self.x_input_sigma_space - eps_estimation * sigma

                denoised = denoised - blur(denoised, k) + blur(latent.to(denoised), k)

                if isinstance(self.sharpness, float):
                    detail_weight = float(self.sharpness) * 0.01
                    neg = detail_weight * blur(denoised, k) + (1 - detail_weight) * denoised
                    denoised = (1 - cond_mark) * denoised + cond_mark * neg

                eps_modified = (self.x_input_sigma_space - denoised) / sigma

                return eps_modified

            return h

        unet.add_block_modifier(block_proc)

        process.sd_model.forge_objects.unet = unet

        return cond, mask


class PreprocessorTileColorFixSharp(PreprocessorTileColorFix):
    def __init__(self):
        super().__init__()
        self.name = 'tile_colorfix+sharp'
        self.slider_2 = PreprocessorParameter(label='Sharpness', value=1.0, minimum=0.0, maximum=2.0, step=0.01, visible=True)

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        self.sharpness = float(kwargs['unit'].threshold_b)
        return super().process_before_every_sampling(process, cond, mask, *args, **kwargs)


add_supported_preprocessor(PreprocessorTile())

add_supported_preprocessor(PreprocessorTileColorFix())

add_supported_preprocessor(PreprocessorTileColorFixSharp())
