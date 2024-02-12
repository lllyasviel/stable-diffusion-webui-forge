import torch
import gradio as gr
import os
import pathlib

from modules import script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules.ui_components import ResizeHandleRow
from modules import shared

from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy
from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.contrib.external_stable3d import StableZero123_Conditioning
from ldm_patched.contrib.external import KSampler, VAEDecode


opStableZero123_Conditioning = StableZero123_Conditioning()
opKSampler = KSampler()
opVAEDecode = VAEDecode()

model_root = os.path.join(models_path, 'z123')
os.makedirs(model_root, exist_ok=True)
model_filenames = []


def update_model_filenames():
    global model_filenames
    model_filenames = [
        pathlib.Path(x).name for x in
        shared.walk_files(model_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])
    ]
    return model_filenames


@torch.inference_mode()
@torch.no_grad()
def predict(filename, width, height, batch_size, elevation, azimuth,
            sampling_seed, sampling_steps, sampling_cfg, sampling_sampler_name, sampling_scheduler, sampling_denoise, input_image):
    filename = os.path.join(model_root, filename)
    model, _, vae, clip_vision = \
        load_checkpoint_guess_config(filename, output_vae=True, output_clip=False, output_clipvision=True)
    init_image = numpy_to_pytorch(input_image)
    positive, negative, latent_image = opStableZero123_Conditioning.encode(
        clip_vision, init_image, vae, width, height, batch_size, elevation, azimuth)
    output_latent = opKSampler.sample(model, sampling_seed, sampling_steps, sampling_cfg,
                                      sampling_sampler_name, sampling_scheduler, positive,
                                      negative, latent_image, sampling_denoise)[0]
    output_pixels = opVAEDecode.decode(vae, output_latent)[0]
    outputs = pytorch_to_numpy(output_pixels)
    return outputs


def on_ui_tabs():
    with gr.Blocks() as model_block:
        with ResizeHandleRow():
            with gr.Column():
                input_image = gr.Image(label='Input Image', source='upload', type='numpy', height=400)

                with gr.Row():
                    filename = gr.Dropdown(label="Zero123 Checkpoint Filename",
                                           choices=model_filenames,
                                           value=model_filenames[0] if len(model_filenames) > 0 else None)
                    refresh_button = ToolButton(value=refresh_symbol, tooltip="Refresh")
                    refresh_button.click(
                        fn=lambda: gr.update(choices=update_model_filenames),
                        inputs=[], outputs=filename)

                width = gr.Slider(label='Width', minimum=16, maximum=8192, step=8, value=256)
                height = gr.Slider(label='Height', minimum=16, maximum=8192, step=8, value=256)
                batch_size = gr.Slider(label='Batch Size', minimum=1, maximum=4096, step=1, value=4)
                elevation = gr.Slider(label='Elevation', minimum=-180.0, maximum=180.0, step=0.001, value=10.0)
                azimuth = gr.Slider(label='Azimuth', minimum=-180.0, maximum=180.0, step=0.001, value=142.0)
                sampling_denoise = gr.Slider(label='Sampling Denoise', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                sampling_steps = gr.Slider(label='Sampling Steps', minimum=1, maximum=10000, step=1, value=20)
                sampling_cfg = gr.Slider(label='CFG Scale', minimum=0.0, maximum=100.0, step=0.1, value=5.0)
                sampling_sampler_name = gr.Radio(label='Sampler Name',
                                                 choices=['euler', 'euler_ancestral', 'heun', 'heunpp2', 'dpm_2',
                                                          'dpm_2_ancestral', 'lms', 'dpm_fast', 'dpm_adaptive',
                                                          'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_sde_gpu',
                                                          'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu',
                                                          'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'lcm', 'ddim',
                                                          'uni_pc', 'uni_pc_bh2'], value='euler')
                sampling_scheduler = gr.Radio(label='Sampling Scheduler',
                                              choices=['normal', 'karras', 'exponential', 'sgm_uniform', 'simple',
                                                       'ddim_uniform'], value='sgm_uniform')
                sampling_seed = gr.Number(label='Seed', value=12345, precision=0)
                generate_button = gr.Button(value="Generate")

                ctrls = [filename, width, height, batch_size, elevation, azimuth, sampling_seed, sampling_steps, sampling_cfg, sampling_sampler_name, sampling_scheduler, sampling_denoise, input_image]

            with gr.Column():
                output_gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain',
                                            visible=True, height=1024, columns=4)

        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery])
    return [(model_block, "Z123", "z123")]


update_model_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
