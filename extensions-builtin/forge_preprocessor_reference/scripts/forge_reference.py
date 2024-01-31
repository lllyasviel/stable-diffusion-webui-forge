import torch

from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from ldm_patched.modules.samplers import sampling_function
import ldm_patched.ldm.modules.attention as attention


def sdp(q, k, v, transformer_options):
    return attention.optimized_attention(q, k, v, heads=transformer_options["n_heads"], mask=None)


class PreprocessorReference(Preprocessor):
    def __init__(self, name, use_attn=True, use_adain=True, priority=0):
        super().__init__()
        self.name = name
        self.use_attn = use_attn
        self.use_adain = use_adain
        self.sorting_priority = priority
        self.tags = ['Reference']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.slider_1 = PreprocessorParameter(label='Style Fidelity', value=0.5, minimum=0.0, maximum=1.0, step=0.01, visible=True)
        self.show_control_mode = False
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = False
        self.do_not_need_model = True

        self.is_recording_style = False
        self.recorded_attn1 = {}
        self.recorded_h = {}

    def process_before_every_sampling(self, process, cond, *args, **kwargs):
        unit = kwargs['unit']
        weight = float(unit.weight)
        style_fidelity = float(unit.threshold_a)
        start_percent = float(unit.guidance_start)
        end_percent = float(unit.guidance_end)

        vae = process.sd_model.forge_objects.vae
        # This is a powerful VAE with integrated memory management, bf16, and tiled fallback.

        latent_image = vae.encode(cond.movedim(1, -1))
        latent_image = process.sd_model.forge_objects.unet.model.latent_format.process_in(latent_image)

        gen_seed = process.seeds[0] + 1
        gen_cpu = torch.Generator().manual_seed(gen_seed)

        unet = process.sd_model.forge_objects.unet.clone()
        sigma_max = unet.model.model_sampling.percent_to_sigma(start_percent)
        sigma_min = unet.model.model_sampling.percent_to_sigma(end_percent)

        self.recorded_attn1 = {}
        self.recorded_h = {}

        def conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
            sigma = timestep[0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return model, x, timestep, uncond, cond, cond_scale, model_options, seed

            self.is_recording_style = True

            xt = latent_image.to(x) + torch.randn(x.size(), dtype=x.dtype, generator=gen_cpu).to(x) * sigma
            sampling_function(model, xt, timestep, uncond, cond, 1, model_options, seed)

            self.is_recording_style = False

            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        def block_proc(h, flag, transformer_options):
            if not self.use_adain:
                return h

            if flag != 'after':
                return h

            location = transformer_options['block']

            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return h

            if self.is_recording_style:
                self.recorded_h[location] = h
            else:
                cond_mark = transformer_options['cond_mark'][:, None, None, None]  # cond is 0
                recorded_h = self.recorded_h[location]
                b = 0

            return h

        def attn1_proc(q, k, v, transformer_options):
            if not self.use_attn:
                return sdp(q, k, v, transformer_options)

            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return sdp(q, k, v, transformer_options)

            location = (transformer_options['block'][0], transformer_options['block'][1],
                        transformer_options['block_index'])

            if self.is_recording_style:
                self.recorded_attn1[location] = (k, v)
            else:
                cond_mark = transformer_options['cond_mark'][:, None, None, None]  # cond is 0
                recorded_attn1 = self.recorded_attn1[location]
                b = 0

            return sdp(q, k, v, transformer_options)

        unet.add_block_modifier(block_proc)
        unet.add_conditioning_modifier(conditioning_modifier)
        unet.set_model_replace_all(attn1_proc, 'attn1')

        process.sd_model.forge_objects.unet = unet

        return


add_supported_preprocessor(PreprocessorReference(
    name='reference_only',
    use_attn=True,
    use_adain=False,
    priority=100
))

add_supported_preprocessor(PreprocessorReference(
    name='reference_adain',
    use_attn=False,
    use_adain=True
))

add_supported_preprocessor(PreprocessorReference(
    name='reference_adain+attn',
    use_attn=True,
    use_adain=True
))
