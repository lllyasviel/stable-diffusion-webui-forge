import spaces

import functools
import os
import shutil
import sys
import git

import gradio as gr
import numpy as np
import torch as torch
from PIL import Image

from gradio_imageslider import ImageSlider

import spaces


import argparse
import os
import logging

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import glob
import json
import cv2

import sys

from geo_models.geowizard_pipeline import DepthNormalEstimationPipeline
from geo_utils.seed_all import seed_all
import matplotlib.pyplot as plt
from geo_utils.de_normalized import align_scale_shift
from geo_utils.depth2normal import *

from diffusers import DiffusionPipeline, DDIMScheduler, AutoencoderKL
from geo_models.unet_2d_condition import UNet2DConditionModel

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

device = spaces.gpu

with spaces.capture_gpu_object() as gpu_object:
    vae = AutoencoderKL.from_pretrained(spaces.convert_root_path(), subfolder='vae')
    scheduler = DDIMScheduler.from_pretrained(spaces.convert_root_path(), subfolder='scheduler')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(spaces.convert_root_path(), subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(spaces.convert_root_path(), subfolder="feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(spaces.convert_root_path(), subfolder="unet")

pipe = DepthNormalEstimationPipeline(vae=vae,
                                     image_encoder=image_encoder,
                                     feature_extractor=feature_extractor,
                                     unet=unet,
                                     scheduler=scheduler)

outputs_dir = "./outputs"

spaces.automatically_move_pipeline_components(pipe)
spaces.automatically_move_to_gpu_when_forward(pipe.vae.encoder, target_model=pipe.vae)
spaces.automatically_move_to_gpu_when_forward(pipe.vae.decoder, target_model=pipe.vae)
spaces.automatically_move_to_gpu_when_forward(pipe.vae.post_quant_conv, target_model=pipe.vae)
# spaces.change_attention_from_diffusers_to_forge(vae)
# spaces.change_attention_from_diffusers_to_forge(unet)
# pipe = pipe.to(device)


@spaces.GPU(gpu_objects=gpu_object, manual_load=True)
def depth_normal(img,
                 denoising_steps,
                 ensemble_size,
                 processing_res,
                 seed,
                 domain):
    seed = int(seed)
    if seed >= 0:
        torch.manual_seed(seed)

    pipe_out = pipe(
        img,
        denoising_steps=denoising_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        batch_size=0,
        domain=domain,
        show_progress_bar=True,
    )

    depth_colored = Image.fromarray(((1. - pipe_out.depth_np) * 255.0).clip(0, 255).astype(np.uint8))
    normal_colored = pipe_out.normal_colored

    return depth_colored, normal_colored


def run_demo():
    custom_theme = gr.themes.Soft(primary_hue="blue").set(
        button_secondary_background_fill="*neutral_100",
        button_secondary_background_fill_hover="*neutral_200")
    custom_css = '''#disp_image {
        text-align: center; /* Horizontally center the content */
    }'''

    _TITLE = '''GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image'''
    _DESCRIPTION = '''
    <div>
    Generate consistent depth and normal from single image. High quality and rich details. (PS: We find the demo running on ZeroGPU output slightly inferior results compared to A100 or 3060 with everything exactly the same.)
    <a style="display:inline-block; margin-left: .5em" href='https://github.com/fuxiao0719/GeoWizard/'><img src='https://img.shields.io/github/stars/fuxiao0719/GeoWizard?style=social' /></a>
    </div>
    '''
    _GPU_ID = 0

    with gr.Blocks(title=_TITLE, theme=custom_theme, css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                input_image = gr.Image(type='pil', image_mode='RGBA', height=320, label='Input image')

                example_folder = os.path.join(spaces.convert_root_path(), "files")
                example_fns = [os.path.join(example_folder, example) for example in os.listdir(example_folder)]
                gr.Examples(
                    examples=example_fns,
                    inputs=[input_image],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=30
                )
            with gr.Column(scale=1):
                with gr.Accordion('Advanced options', open=True):
                    with gr.Column():
                        domain = gr.Radio(
                            [
                                ("Outdoor", "outdoor"),
                                ("Indoor", "indoor"),
                                ("Object", "object"),
                            ],
                            label="Data Type (Must Select One matches your image)",
                            value="indoor",
                        )
                        denoising_steps = gr.Slider(
                            label="Number of denoising steps (More steps, better quality)",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=10,
                        )
                        ensemble_size = gr.Slider(
                            label="Ensemble size (More steps, higher accuracy)",
                            minimum=1,
                            maximum=15,
                            step=1,
                            value=3,
                        )
                        seed = gr.Number(0, label='Random Seed. Negative values for not specifying')

                        processing_res = gr.Radio(
                            [
                                ("Native", 0),
                                ("Recommended", 768),
                            ],
                            label="Processing resolution",
                            value=768,
                        )

                run_btn = gr.Button('Generate', variant='primary', interactive=True)
        with gr.Row():
            with gr.Column():
                depth = gr.Image(interactive=False, show_label=False)
            with gr.Column():
                normal = gr.Image(interactive=False, show_label=False)

        run_btn.click(fn=depth_normal,
                      inputs=[input_image, denoising_steps,
                              ensemble_size,
                              processing_res,
                              seed,
                              domain],
                      outputs=[depth, normal]
                      )
        return demo


demo = run_demo()
if __name__ == '__main__':
    demo.queue().launch(share=True, max_threads=80)
