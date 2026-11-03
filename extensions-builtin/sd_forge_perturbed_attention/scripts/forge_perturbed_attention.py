import gradio as gr

from modules import scripts
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from backend.patcher.base import set_model_options_patch_replace
from backend.sampling.sampling_function import calc_cond_uncond_batch
from modules.ui_components import InputAccordion


class PerturbedAttentionGuidanceForForge(scripts.Script):
    sorting_priority = 13

    attenuated_scale = 3.0
    doPAG = True

    def title(self):
        return "PerturbedAttentionGuidance Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            with gr.Row():
                scale = gr.Slider(label='Scale', minimum=0.0, maximum=100.0, step=0.1, value=3.0)
                attenuation = gr.Slider(label='Attenuation (linear, % of scale)', minimum=0.0, maximum=100.0, step=0.1, value=0.0)
            with gr.Row():
                start_step = gr.Slider(label='Start step', minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                end_step = gr.Slider(label='End step', minimum=0.0, maximum=1.0, step=0.01, value=1.0)

        self.infotext_fields = [
            (enabled, lambda d: d.get("pagi_enabled", False)),
            (scale,         "pagi_scale"),
            (attenuation,   "pagi_attenuation"),
            (start_step,    "pagi_start_step"),
            (end_step,      "pagi_end_step"),
        ]

        return enabled, scale, attenuation, start_step, end_step

    def denoiser_callback(self, params):
        thisStep = (params.sampling_step) / (params.total_sampling_steps - 1)
        
        if thisStep >= PerturbedAttentionGuidanceForForge.PAG_start and thisStep <= PerturbedAttentionGuidanceForForge.PAG_end:
            PerturbedAttentionGuidanceForForge.doPAG = True
        else:
            PerturbedAttentionGuidanceForForge.doPAG = False

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, scale, attenuation, start_step, end_step = script_args

        if not enabled:
            return

        PerturbedAttentionGuidanceForForge.scale = scale
        PerturbedAttentionGuidanceForForge.PAG_start = start_step
        PerturbedAttentionGuidanceForForge.PAG_end = end_step
        on_cfg_denoiser(self.denoiser_callback)

        unet = p.sd_model.forge_objects.unet.clone()

        def attn_proc(q, k, v, to):
            return v

        def post_cfg_function(args):
            denoised = args["denoised"]

            if PerturbedAttentionGuidanceForForge.scale <= 0.0:
                return denoised

            if not PerturbedAttentionGuidanceForForge.doPAG:
                return denoised
            
            model, cond_denoised, cond, sigma, x, options = \
                args["model"], args["cond_denoised"], args["cond"], args["sigma"], args["input"], args["model_options"].copy()
            new_options = set_model_options_patch_replace(options, attn_proc, "attn1", "middle", 0)

            degraded, _ = calc_cond_uncond_batch(model, cond, None, x, sigma, new_options)

            result = denoised + (cond_denoised - degraded) * PerturbedAttentionGuidanceForForge.scale
            PerturbedAttentionGuidanceForForge.scale -= scale * attenuation / 100.0

            return result

        unet.set_model_sampler_post_cfg_function(post_cfg_function)

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            pagi_enabled     = enabled,
            pagi_scale       = scale,
            pagi_attenuation = attenuation,
            pagi_start_step  = start_step,
            pagi_end_step    = end_step,
        ))

        return

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return
