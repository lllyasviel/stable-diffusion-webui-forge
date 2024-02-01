# Use --show-controlnet-example to see this extension.

import cv2
import gradio as gr
import torch

from modules import scripts
from modules.shared_cmd_options import cmd_opts
from modules_forge.shared import supported_preprocessors
from modules.modelloader import load_file_from_url
from ldm_patched.modules.controlnet import load_controlnet
from modules_forge.controlnet import apply_controlnet_advanced
from modules_forge.forge_util import numpy_to_pytorch
from modules_forge.shared import controlnet_dir


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

        # controlnet_canny_path = load_file_from_url(
        #     url='https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_canny_256lora.safetensors',
        #     model_dir=model_dir,
        #     file_name='sai_xl_canny_256lora.safetensors'
        # )
        controlnet_canny_path = load_file_from_url(
            url='https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/control_v11p_sd15_canny_fp16.safetensors',
            model_dir=controlnet_dir,
            file_name='control_v11p_sd15_canny_fp16.safetensors'
        )
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
        batch_size = p.batch_size

        preprocessor = supported_preprocessors['canny']

        # detect control at certain resolution
        control_image = preprocessor(
            input_image, resolution=512, slider_1=100, slider_2=200, slider_3=None)

        # here we just use nearest neighbour to align input shape.
        # You may want crop and resize, or crop and fill, or others.
        control_image = cv2.resize(
            control_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # Output preprocessor result. Now called every sampling. Cache in your own way.
        p.extra_result_images.append(control_image)

        print('Preprocessor Canny finished.')

        control_image_bchw = numpy_to_pytorch(control_image).movedim(-1, 1)

        unet = p.sd_model.forge_objects.unet

        # Unet has input, middle, output blocks, and we can give different weights
        # to each layers in all blocks.
        # Below is an example for stronger control in middle block.
        # This is helpful for some high-res fix passes. (p.is_hr_pass)
        positive_advanced_weighting = {
            'input': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            'middle': [1.0],
            'output': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        }
        negative_advanced_weighting = {
            'input': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25],
            'middle': [1.05],
            'output': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25]
        }

        # The advanced_frame_weighting is a weight applied to each image in a batch.
        # The length of this list must be same with batch size
        # For example, if batch size is 5, the below list is [0.2, 0.4, 0.6, 0.8, 1.0]
        # If you view the 5 images as 5 frames in a video, this will lead to
        # progressively stronger control over time.
        advanced_frame_weighting = [float(i + 1) / float(batch_size) for i in range(batch_size)]

        # The advanced_sigma_weighting allows you to dynamically compute control
        # weights given diffusion timestep (sigma).
        # For example below code can softly make beginning steps stronger than ending steps.
        sigma_max = unet.model.model_sampling.sigma_max
        sigma_min = unet.model.model_sampling.sigma_min
        advanced_sigma_weighting = lambda s: (s - sigma_min) / (sigma_max - sigma_min)

        # You can even input a tensor to mask all control injections
        # The mask will be automatically resized during inference in UNet.
        # The size should be B 1 H W and the H and W are not important
        # because they will be resized automatically
        advanced_mask_weighting = torch.ones(size=(1, 1, 512, 512))

        # But in this simple example we do not use them
        positive_advanced_weighting = None
        negative_advanced_weighting = None
        advanced_frame_weighting = None
        advanced_sigma_weighting = None
        advanced_mask_weighting = None

        unet = apply_controlnet_advanced(unet=unet, controlnet=self.model, image_bchw=control_image_bchw,
                                         strength=0.6, start_percent=0.0, end_percent=0.8,
                                         positive_advanced_weighting=positive_advanced_weighting,
                                         negative_advanced_weighting=negative_advanced_weighting,
                                         advanced_frame_weighting=advanced_frame_weighting,
                                         advanced_sigma_weighting=advanced_sigma_weighting,
                                         advanced_mask_weighting=advanced_mask_weighting)

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
