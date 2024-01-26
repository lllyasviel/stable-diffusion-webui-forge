import gradio as gr
import os
import pathlib

from modules import scripts, script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules import shared
from modules_forge.gradio_compile import gradio_compile

from ldm_patched.contrib.external_video_model import ImageOnlyCheckpointLoader, VideoLinearCFGGuidance, SVD_img2vid_Conditioning
from ldm_patched.contrib.external import KSampler, VAEDecode


ps = []
ps += gradio_compile(SVD_img2vid_Conditioning.INPUT_TYPES(), prefix='')
ps += gradio_compile(KSampler.INPUT_TYPES(), prefix='sampling')
ps += gradio_compile(VideoLinearCFGGuidance.INPUT_TYPES(), prefix='guidance')
print(', '.join(ps))

svd_root = os.path.join(models_path, 'svd')
os.makedirs(svd_root, exist_ok=True)
svd_filenames = []


def update_svd_filenames():
    global svd_filenames
    svd_filenames = [pathlib.Path(x).name for x in shared.walk_files(svd_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])]
    return svd_filenames


class ForgeSVD(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "SVD"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as svd_block:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    filename = gr.Dropdown(label="SVD Checkpoint Filename",
                                           choices=svd_filenames,
                                           value=svd_filenames[0] if len(svd_filenames) > 0 else None)
                    refresh_button = ToolButton(value=refresh_symbol, tooltip="Refresh")
                    refresh_button.click(
                        fn=lambda: gr.update(choices=update_svd_filenames),
                        inputs=[], outputs=filename)

                a = 0
                width = gr.Slider(label='Width', minimum=16, maximum=8192, step=8, value=1024)
                height = gr.Slider(label='Height', minimum=16, maximum=8192, step=8, value=576)
                video_frames = gr.Slider(label='Video Frames', minimum=1, maximum=4096, step=1, value=14)
                motion_bucket_id = gr.Slider(label='Motion Bucket Id', minimum=1, maximum=1023, step=1, value=127)
                fps = gr.Slider(label='Fps', minimum=1, maximum=1024, step=1, value=6)
                augmentation_level = gr.Slider(label='Augmentation Level', minimum=0.0, maximum=10.0, step=0.01,
                                               value=0.0)
                sampling_seed = gr.Slider(label='Sampling Seed', minimum=0, maximum=18446744073709551615, step=1,
                                          value=0)
                sampling_steps = gr.Slider(label='Sampling Steps', minimum=1, maximum=10000, step=1, value=20)
                sampling_cfg = gr.Slider(label='Sampling Cfg', minimum=0.0, maximum=100.0, step=0.1, value=8.0)
                sampling_sampler_name = gr.Radio(label='Sampling Sampler Name',
                                                 choices=['euler', 'euler_ancestral', 'heun', 'heunpp2', 'dpm_2',
                                                          'dpm_2_ancestral', 'lms', 'dpm_fast', 'dpm_adaptive',
                                                          'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_sde_gpu',
                                                          'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu',
                                                          'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'lcm', 'ddim',
                                                          'uni_pc', 'uni_pc_bh2'], value='euler')
                sampling_scheduler = gr.Radio(label='Sampling Scheduler',
                                              choices=['normal', 'karras', 'exponential', 'sgm_uniform', 'simple',
                                                       'ddim_uniform'], value='normal')
                sampling_denoise = gr.Slider(label='Sampling Denoise', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                guidance_min_cfg = gr.Slider(label='Guidance Min Cfg', minimum=0.0, maximum=100.0, step=0.5, value=1.0)
                clip_vision, init_image, vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level, sampling_model, sampling_seed, sampling_steps, sampling_cfg, sampling_sampler_name, sampling_scheduler, sampling_positive, sampling_negative, sampling_latent_image, sampling_denoise, guidance_model, guidance_min_cfg
                generate_button = gr.Button(value="Generate")

            with gr.Column():
                output_gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain',
                                            visible=True, height=1024, columns=4)

    return [(svd_block, "SVD", "svd")]


update_svd_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
