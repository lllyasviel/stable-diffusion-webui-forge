import gradio as gr

from modules import scripts
from backend.patcher.base import set_model_options_patch_replace
from backend.sampling.sampling_function import calc_cond_uncond_batch


class PerturbedAttentionGuidanceForForge(scripts.Script):
    sorting_priority = 13

    def title(self):
        return "PerturbedAttentionGuidance Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            scale = gr.Slider(label='Scale', minimum=0.0, maximum=100.0, step=0.1, value=3.0)

        return enabled, scale

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, scale = script_args

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet.clone()

        def attn_proc(q, k, v, to):
            return v

        def post_cfg_function(args):
            model, cond_denoised, cond, denoised, sigma, x = \
                args["model"], args["cond_denoised"], args["cond"], args["denoised"], args["sigma"], args["input"]

            new_options = set_model_options_patch_replace(args["model_options"], attn_proc, "attn1", "middle", 0)

            if scale == 0:
                return denoised

            degraded, _ = calc_cond_uncond_batch(model, cond, None, x, sigma, new_options)

            return denoised + (cond_denoised - degraded) * scale

        unet.set_model_sampler_post_cfg_function(post_cfg_function)

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            PerturbedAttentionGuidance_enabled=enabled,
            PerturbedAttentionGuidance_scale=scale,
        ))

        return
