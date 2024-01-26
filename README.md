# This is a Private Project

Currently, we are only sending invitations to people who may be interested in development of this project.

Please do not share codes or info from this project to public.

If you see this, please join our private Discord server for discussion: https://discord.gg/eTfuzT2z

# Stable Diffusion Web UI Forge

Stable Diffusion Web UI Forge is a platform on top of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to make development easier, and optimize the speed and resource consumption.

The name "Forge" is inspired from "Minecraft Forge". This project will become SD WebUI's Forge.

Forge will give you:

1. Improved optimization. (Fastest speed and minimal memory use among all alternative software.)
2. Patchable UNet and CLIP objects. (Developer-friendly platform.)

# Improved Optimization

I tested with several devices, and this is a typical result from 8GB VRAM (3070ti laptop) with SDXL.

**This is WebUI:**

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/c32baedd-500b-408f-8cfb-ed4570c883bd)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/cb6098de-f2d4-4b25-9566-df4302dda396)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/5447e8b7-f3ca-4003-9961-02027c8181e8)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/f3cb57d9-ac7a-4667-8b3f-303139e38afa)

(average about 7.4GB/8GB, peak at about 7.9GB/8GB)

**This is WebUI Forge:**

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/0c45cd98-0b14-42c3-9556-28e48d4d5fa0)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/3a71f5d4-39e5-4ab1-81cf-8eaa790a2dc8)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/65fbb4a5-ee73-4bb9-9c5f-8a958cd9674d)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/76f181a1-c5fb-4323-a6cc-b6308a45587e)

(average and peak are all 6.3GB/8GB)

Also, you can see that Forge does not change WebUI results. Installing Forge is not a seed breaking change. 

We do not change any UI. But you will see the version of Forge here

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/93fdbccf-2f9b-4d45-ad81-c7c4106a357b)

"f0.0.1v1.7.0" means WebUI 1.7.0 with Forge 0.0.1

### Changes

Forge removes all WebUI's codes related to speed and memory optimization and reworked everything. 

All previous cmd flags like medvram, lowvram, medvram-sdxl, precision full, no half, no half vae, attention_xxx, upcast unet, ... are all REMOVED. Adding these flags will not cause error but they will not do anything now. **We highly encourage Forge users to remove all cmd flags and let Forge to decide how to load models.**

Currently, the behaviors is:

"When loading a model to GPU, Forge will decide whether to load the entire model, or to load separated parts of the model. Then, when loading another model, Forge will try best to unload the previous model."

**The only one flag that you may still need** is `--disable-offload-from-vram`, to change the above behavior to

"When loading a model to GPU, Forge will decide whether to load the entire model, or to load separated parts of the model. Then, when loading another model, Forge will try best to keep the previous model in GPU without unloading it."

You should `--disable-offload-from-vram` when and only when you have more than 20GB GPU memory, or when you are on MAC MPS.

If you really want to play with cmd flags, you can additionally control the GPU with:

(extreme VRAM cases)

    --always-gpu
    --always-cpu

(rare attention cases)

    --attention-split
    --attention-quad
    --attention-pytorch
    --disable-xformers
    --disable-attention-upcast

(float point type)

    --all-in-fp32
    --all-in-fp16
    --unet-in-bf16
    --unet-in-fp16
    --unet-in-fp8-e4m3fn
    --unet-in-fp8-e5m2
    --vae-in-fp16
    --vae-in-fp32
    --vae-in-bf16
    --clip-in-fp8-e4m3fn
    --clip-in-fp8-e5m2
    --clip-in-fp16
    --clip-in-fp32

(rare platforms)

    --directml
    --disable-ipex-hijack
    --pytorch-deterministic

Again, Forge do not recommend users to use any cmd flags unless you are very sure that you really need these.

# Patchable UNet

Now developing an extension is super simple. We finally have patchable UNet.

Below is using one single file with 80 lines of codes to support FreeU:

`extensions-builtin/sd_forge_freeu/scripts/forge_freeu.py`

```python
import torch
import gradio as gr
from modules import scripts


def Fourier_filter(x, threshold, scale):
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)
    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered.to(x.dtype)


def set_freeu_v2_patch(model, b1, b2, s1, s2):
    model_channels = model.model.model_config.unet_config["model_channels"]
    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}

    def output_block_patch(h, hsp, *args, **kwargs):
        scale = scale_dict.get(h.shape[1], None)
        if scale is not None:
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / \
                          (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
            h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)
            hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
        return h, hsp

    m = model.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


class FreeUForForge(scripts.Script):
    def title(self):
        return "FreeU Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            freeu_enabled = gr.Checkbox(label='Enabled', value=False)
            freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
            freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
            freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
            freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)

        return freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2

    def process_batch(self, p, *script_args, **kwargs):
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2 = script_args

        if not freeu_enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = set_freeu_v2_patch(unet, freeu_b1, freeu_b2, freeu_s1, freeu_s2)

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            freeu_enabled=freeu_enabled,
            freeu_b1=freeu_b1,
            freeu_b2=freeu_b2,
            freeu_s1=freeu_s1,
            freeu_s2=freeu_s2,
        ))

        return
```

It looks like this:

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/a7798cf2-057c-43e0-883a-5f8643af8529)

Similar components like HyperTile, KohyaHighResFix, SAG, can all be implemented within 100 lines of codes (see also the codes).

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/e2fc1b73-e6ee-405e-864c-c67afd92a1db)

ControlNets can finally be called by different extensions. (80% codes of ControlNet can be removed now, will start soon)

Implementing Stable Video Diffusion and Zero123 are also super simple now (see also the codes). 

*Stable Video Diffusion:*

`extensions-builtin/sd_forge_freeu/scripts/forge_svd.py`

```python
import torch
import gradio as gr
import os
import pathlib

from modules import script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules import shared

from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy
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
    return outputs


def on_ui_tabs():
    with gr.Blocks() as svd_block:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label='Input Image', source='upload', type='numpy', height=400)

                with gr.Row():
                    filename = gr.Dropdown(label="SVD Checkpoint Filename",
                                           choices=svd_filenames,
                                           value=svd_filenames[0] if len(svd_filenames) > 0 else None)
                    refresh_button = ToolButton(value=refresh_symbol, tooltip="Refresh")
                    refresh_button.click(
                        fn=lambda: gr.update(choices=update_svd_filenames),
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
                output_gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain',
                                            visible=True, height=1024, columns=4)

        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery])
    return [(svd_block, "SVD", "svd")]


update_svd_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
```

Note that although the above codes look like independent codes, they actually will automatically offload/unload any other models. For example, below is me opening webui, load SDXL, generated an image, then go to SVD, then generated image frames. You can see that the GPU memory is perfectly managed and the SDXL is moved to RAM then SVD is moved to GPU. 

Note that this management is fully automatic. This makes writing extensions super simple.

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/ac7ed152-cd33-4645-94af-4c43bb8c3d88)


![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/cdcb23ad-02dc-4e39-be74-98e927550ef6)


Similarly, Zero123:


