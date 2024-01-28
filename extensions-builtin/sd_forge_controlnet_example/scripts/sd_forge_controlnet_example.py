# Use --show-controlnet-example to see this extension.

import os
import cv2
import gradio as gr

from modules import scripts
from modules.shared_cmd_options import cmd_opts
from modules.paths import models_path
from modules.modelloader import load_file_from_url
from ldm_patched.modules.controlnet import load_controlnet


class ControlNetExampleForge(scripts.Script):
    def title(self):
        return "ControlNet Example for Developers"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML('This is an example controlnet extension for developers.')
            gr.HTML('You see this extension because you used --show-controlnet-example')
            input_image = gr.Image(source='upload', type='numpy')
            funny_slider = gr.Slider(label='This slider does nothing. It just shows you how to transfer parameters.',
                                     minimum=0.0, maximum=1.0, value=0.5)

        return input_image, funny_slider

    def process(self, p, *script_args, **kwargs):
        input_image, funny_slider = script_args

        # This slider does nothing. It just shows you how to transfer parameters.
        del funny_slider

        if input_image is None:
            return

        model_dir = os.path.join(models_path, 'ControlNet')
        os.makedirs(model_dir, exist_ok=True)
        controlnet_canny_path = load_file_from_url(
            url='https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/control_v11p_sd15_canny_fp16.safetensors',
            model_dir=model_dir,
            file_name='control_v11p_sd15_canny_fp16.safetensors'
        )
        print('The model [control_v11p_sd15_canny_fp16.safetensors] download finished.')

        controlnet = load_controlnet(controlnet_canny_path)
        print('Controlnet loaded.')

        return

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        input_image, funny_slider = script_args

        if input_image is None:
            return

        B, C, H, W = kwargs['noise'].shape  # latent_shape
        height = H * 8
        width = W * 8

        input_image = cv2.resize(input_image, (height, width))
        canny_image = cv2.Canny(input_image, 100, 200)

        # Output preprocessor result. Now called every sampling. Cache in your own way.
        p.extra_result_images.append(canny_image)

        print('Preprocessor Canny finished.')

        return


# Use --show-controlnet-example to see this extension.
if not cmd_opts.show_controlnet_example:
    del ControlNetExampleForge
