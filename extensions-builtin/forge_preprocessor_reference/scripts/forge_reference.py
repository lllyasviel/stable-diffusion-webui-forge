import torch

from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from ldm_patched.modules.samplers import sampling_function


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

            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return h

            if self.is_recording_style:
                a = 0
            else:
                b = 0

            return h

        def attn1_proc(q, k, v, transformer_options):
            if not self.use_attn:
                return q, k, v

            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return q, k, v

            if self.is_recording_style:
                a = 0
            else:
                b = 0

            return q, k, v

        def attn1_output_proc(h, transformer_options):
            if not self.use_attn:
                return h

            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return h

            if self.is_recording_style:
                a = 0
            else:
                b = 0

            return h

        unet.add_block_modifier(block_proc)
        unet.add_conditioning_modifier(conditioning_modifier)
        unet.set_model_attn1_patch(attn1_proc)
        unet.set_model_attn1_output_patch(attn1_output_proc)

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
