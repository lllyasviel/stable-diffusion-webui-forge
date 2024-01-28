# Use --show-controlnet-example to see this extension.

import os
import cv2
import gradio as gr

from modules import scripts
from modules.shared_cmd_options import cmd_opts
from modules.paths import models_path
from modules.modelloader import load_file_from_url
from ldm_patched.modules.controlnet import load_controlnet
from modules_forge.controlnet import apply_controlnet_advanced
from modules_forge.forge_util import pytorch_to_numpy, numpy_to_pytorch


class ControlNetExampleForge(scripts.Script):
    model = None

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
            url='https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_canny_256lora.safetensors',
            model_dir=model_dir,
            file_name='sai_xl_canny_256lora.safetensors'
        )
        # controlnet_canny_path = load_file_from_url(
        #     url='https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/control_v11p_sd15_canny_fp16.safetensors',
        #     model_dir=model_dir,
        #     file_name='control_v11p_sd15_canny_fp16.safetensors'
        # )
        print('The model [control_v11p_sd15_canny_fp16.safetensors] download finished.')

        self.model = load_controlnet(controlnet_canny_path)
        print('Controlnet loaded.')

        return

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        input_image, funny_slider = script_args

        if input_image is None or self.model is None:
            return

        B, C, H, W = kwargs['noise'].shape  # latent_shape
        height = H * 8
        width = W * 8

        input_image = cv2.resize(input_image, (width, height))
        canny_image = cv2.cvtColor(cv2.Canny(input_image, 100, 200), cv2.COLOR_GRAY2RGB)

        # Output preprocessor result. Now called every sampling. Cache in your own way.
        p.extra_result_images.append(canny_image)

        print('Preprocessor Canny finished.')

        control_image = numpy_to_pytorch(canny_image)

        unet = p.sd_model.forge_objects.unet

        unet = apply_controlnet_advanced(unet=unet, controlnet=self.model, image_bhwc=control_image,
                                         strength=0.6, start_percent=0.0, end_percent=0.8,
                                         positive_advanced_weighting=None, negative_advanced_weighting=None)

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            controlnet_info='You should see these texts below output images!',
        ))

        return


# Use --show-controlnet-example to see this extension.
if not cmd_opts.show_controlnet_example:
    del ControlNetExampleForge
