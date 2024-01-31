import torch

from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor


class PreprocessorReference(Preprocessor):
    def __init__(self, name, use_attn=True, use_adain=True, priority=0):
        super().__init__()
        self.name = name
        self.use_attn = use_attn
        self.use_adain = use_adain
        self.sorting_priority = priority
        self.tags = ['Reference']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.slider_1 = PreprocessorParameter(label='Style Fidelity', value=0.5, minimum=0.0, maximum=1.0, step=0.01)
        self.show_control_mode = False
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = False
        self.do_not_need_model = True

    def process_before_every_sampling(self, process, cond, *args, **kwargs):
        unit = kwargs['unit']
        weight = float(unit.weight)
        style_fidelity = float(unit.threshold_a)
        start_percent = float(unit.guidance_start)
        end_percent = float(unit.guidance_end)

        unet = process.sd_model.forge_objects.unet.clone()
        sigma_max = unet.model_sampling.percent_to_sigma(start_percent)
        sigma_min = unet.model_sampling.percent_to_sigma(end_percent)

        def conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
            sigma = timestep[0].item()
            if not (sigma_min < sigma < sigma_max):
                return model, x, timestep, uncond, cond, cond_scale, model_options, seed

            a = 0

            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        def block_proc(h, flag, transformer_options):
            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min < sigma < sigma_max):
                return h

            a = 0

            return h

        def attn1_proc(q, k, v, transformer_options):
            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min < sigma < sigma_max):
                return q, k, k

            a = 0

            return q, k, k

        unet.add_block_modifier(block_proc)
        unet.add_conditioning_modifier(conditioning_modifier)
        unet.set_model_attn1_patch(attn1_proc)

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
