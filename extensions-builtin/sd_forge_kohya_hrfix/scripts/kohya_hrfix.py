import gradio as gr

from modules import scripts
from ldm_patched.contrib.external_model_downscale import PatchModelAddDownscale


opPatchModelAddDownscale = PatchModelAddDownscale()


class KohyaHRFixForForge(scripts.Script):
    sorting_priority = 14

    def title(self):
        return "Kohya HRFix Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        upscale_methods = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"]
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            block_number = gr.Slider(label='Block Number', value=3, minimum=1, maximum=32, step=1)
            downscale_factor = gr.Slider(label='Downscale Factor', value=2.0, minimum=0.1, maximum=9.0, step=0.001)
            start_percent = gr.Slider(label='Start Percent', value=0.0, minimum=0.0, maximum=1.0, step=0.001)
            end_percent = gr.Slider(label='End Percent', value=0.35, minimum=0.0, maximum=1.0, step=0.001)
            downscale_after_skip = gr.Checkbox(label='Downscale After Skip', value=True)
            downscale_method = gr.Radio(label='Downscale Method', choices=upscale_methods, value=upscale_methods[0])
            upscale_method = gr.Radio(label='Upscale Method', choices=upscale_methods, value=upscale_methods[0])

        return enabled, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method = script_args
        block_number = int(block_number)

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = opPatchModelAddDownscale.patch(unet, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method)[0]

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            kohya_hrfix_enabled=enabled,
            kohya_hrfix_block_number=block_number,
            kohya_hrfix_downscale_factor=downscale_factor,
            kohya_hrfix_start_percent=start_percent,
            kohya_hrfix_end_percent=end_percent,
            kohya_hrfix_downscale_after_skip=downscale_after_skip,
            kohya_hrfix_downscale_method=downscale_method,
            kohya_hrfix_upscale_method=upscale_method,
        ))

        return
