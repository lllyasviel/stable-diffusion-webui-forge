import torch
import gradio as gr
import os
import pathlib

import modules.infotext_utils as parameters_copypaste
from modules import script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules.ui_components import ResizeHandleRow 
from modules import shared

from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy, write_images_to_mp4
from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.contrib.external_video_model import VideoLinearCFGGuidance, SVD_img2vid_Conditioning
from ldm_patched.contrib.external import KSampler, VAEDecode


opVideoLinearCFGGuidance = VideoLinearCFGGuidance()
opSVD_img2vid_Conditioning = SVD_img2vid_Conditioning()
opKSampler = KSampler()
opVAEDecode = VAEDecode()

svd_root = os.path.join(models_path, 'svd')
os.makedirs(svd_root, exist_ok=True)
svd_filenames = []


def update_svd_filenames():
    global svd_filenames
    svd_filenames = [
        pathlib.Path(x).name for x in
        shared.walk_files(svd_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])
    ]
    return svd_filenames


@torch.inference_mode()
@torch.no_grad()
def predict(filename, width, height, video_frames, motion_bucket_id, fps, augmentation_level,
            sampling_seed, sampling_steps, sampling_cfg, sampling_sampler_name, sampling_scheduler,
            sampling_denoise, guidance_min_cfg, input_image):
    filename = os.path.join(svd_root, filename)
    model_raw, _, vae, clip_vision = \
        load_checkpoint_guess_config(filename, output_vae=True, output_clip=False, output_clipvision=True)
    model = opVideoLinearCFGGuidance.patch(model_raw, guidance_min_cfg)[0]
    init_image = numpy_to_pytorch(input_image)
    positive, negative, latent_image = opSVD_img2vid_Conditioning.encode(
        clip_vision, init_image, vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level)
    output_latent = opKSampler.sample(model, sampling_seed, sampling_steps, sampling_cfg,
                                      sampling_sampler_name, sampling_scheduler, positive,
                                      negative, latent_image, sampling_denoise)[0]
    output_pixels = opVAEDecode.decode(vae, output_latent)[0]
    outputs = pytorch_to_numpy(output_pixels)

    video_filename = write_images_to_mp4(outputs, fps=fps)

    return outputs, video_filename


def on_ui_tabs():
    with gr.Blocks() as svd_block:
        with ResizeHandleRow():
            with gr.Column():
                input_image = gr.Image(label='Input Image', source='upload', type='numpy', height=400)

                with gr.Row():
                    filename = gr.Dropdown(label="SVD Checkpoint Filename",
                                           choices=svd_filenames,
                                           value=svd_filenames[0] if len(svd_filenames) > 0 else None)
                    refresh_button = ToolButton(value=refresh_symbol, tooltip="Refresh")
                    refresh_button.click(
                        fn=lambda: gr.update(choices=update_svd_filenames()),
                        inputs=[], outputs=filename)

                width = gr.Slider(label='Width', minimum=16, maximum=8192, step=8, value=1024)
                height = gr.Slider(label='Height', minimum=16, maximum=8192, step=8, value=576)
                video_frames = gr.Slider(label='Video Frames', minimum=1, maximum=4096, step=1, value=14)
                motion_bucket_id = gr.Slider(label='Motion Bucket Id', minimum=1, maximum=1023, step=1, value=127)
                fps = gr.Slider(label='Fps', minimum=1, maximum=1024, step=1, value=6)
                augmentation_level = gr.Slider(label='Augmentation Level', minimum=0.0, maximum=10.0, step=0.01,
                                               value=0.0)
                sampling_steps = gr.Slider(label='Sampling Steps', minimum=1, maximum=200, step=1, value=20)
                sampling_cfg = gr.Slider(label='CFG Scale', minimum=0.0, maximum=50.0, step=0.1, value=2.5)
                sampling_denoise = gr.Slider(label='Sampling Denoise', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                guidance_min_cfg = gr.Slider(label='Guidance Min Cfg', minimum=0.0, maximum=100.0, step=0.5, value=1.0)
                sampling_sampler_name = gr.Radio(label='Sampler Name',
                                                 choices=['euler', 'euler_ancestral', 'heun', 'heunpp2', 'dpm_2',
                                                          'dpm_2_ancestral', 'lms', 'dpm_fast', 'dpm_adaptive',
                                                          'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_sde_gpu',
                                                          'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu',
                                                          'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'lcm', 'ddim',
                                                          'uni_pc', 'uni_pc_bh2'], value='euler')
                sampling_scheduler = gr.Radio(label='Scheduler',
                                              choices=['normal', 'karras', 'exponential', 'sgm_uniform', 'simple',
                                                       'ddim_uniform'], value='karras')
                sampling_seed = gr.Number(label='Seed', value=12345, precision=0)

                generate_button = gr.Button(value="Generate")

                ctrls = [filename, width, height, video_frames, motion_bucket_id, fps, augmentation_level,
                         sampling_seed, sampling_steps, sampling_cfg, sampling_sampler_name, sampling_scheduler,
                         sampling_denoise, guidance_min_cfg, input_image]

            with gr.Column():
                output_video = gr.Video(autoplay=True)
                output_gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain',
                                            visible=True, height=1024, columns=4)

        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery, output_video])
        PasteField = parameters_copypaste.PasteField
        paste_fields = [
            PasteField(width, "Size-1", api="width"),
            PasteField(height, "Size-2", api="height"),
        ]
        parameters_copypaste.add_paste_fields("svd", init_img=input_image, fields=paste_fields)
    return [(svd_block, "SVD", "svd")]


update_svd_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
