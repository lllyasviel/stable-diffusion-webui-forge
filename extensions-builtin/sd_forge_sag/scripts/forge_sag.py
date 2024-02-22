import gradio as gr

from modules import scripts
from ldm_patched.contrib.external_sag import SelfAttentionGuidance


opSelfAttentionGuidance = SelfAttentionGuidance()


class SAGForForge(scripts.Script):
    sorting_priority = 12.5

    def title(self):
        return "SelfAttentionGuidance Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            scale = gr.Slider(label='Scale', minimum=-2.0, maximum=5.0, step=0.01, value=0.5)
            blur_sigma = gr.Slider(label='Blur Sigma', minimum=0.0, maximum=10.0, step=0.01, value=2.0)

        return enabled, scale, blur_sigma

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, scale, blur_sigma = script_args

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = opSelfAttentionGuidance.patch(unet, scale, blur_sigma)[0]

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            sag_enabled=enabled,
            sag_scale=scale,
            sag_blur_sigma=blur_sigma
        ))

        return
