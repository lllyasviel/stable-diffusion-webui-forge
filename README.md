# Stable Diffusion Web UI Forge

Stable Diffusion Web UI Forge is a platform on top of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to make development easier, optimize resource management, and speed up inference.

The name "Forge" is inspired from "Minecraft Forge". This project is aimed at becoming SD WebUI's Forge.

Compared to original WebUI (for SDXL inference at 1024px), you can expect the below speed-ups:

1. If you use common GPU like 8GB vram, you can expect to get about **30~45% speed up** in inference speed (it/s), the GPU memory peak (in task manager) will drop about 700MB to 1.3GB, the maximum diffusion resolution (that will not OOM) will increase about 2x to 3x, and the maximum diffusion batch size (that will not OOM) will increase about 4x to 6x.

2. If you use less powerful GPU like 6GB vram, you can expect to get about **60~75% speed up** in inference speed (it/s), the GPU memory peak (in task manager) will drop about 800MB to 1.5GB, the maximum diffusion resolution (that will not OOM) will increase about 3x, and the maximum diffusion batch size (that will not OOM) will increase about 4x.

3. If you use powerful GPU like 4090 with 24GB vram, you can expect to get about **3~6% speed up** in inference speed (it/s), the GPU memory peak (in task manager) will drop about 1GB to 1.4GB, the maximum diffusion resolution (that will not OOM) will increase about 1.6x, and the maximum diffusion batch size (that will not OOM) will increase about 2x.

4. If you use ControlNet for SDXL, the maximum ControlNet count (that will not OOM) will increase about 2x, the speed with SDXL+ControlNet will **speed up about 30~45%**.

Another very important change that Forge brings is **Unet Patcher**. Using Unet Patcher, methods like Self-Attention Guidance, Kohya High Res Fix, FreeU, StyleAlign, Hypertile can all be implemented in about 100 lines of codes. 

Thanks to Unet Patcher, many new things are possible now and supported in Forge, including SVD, Z123, masked Ip-adapter, masked controlnet, photomaker, etc.

**No need to monkey patch UNet and conflict other extensions anymore!**

# Installing Forge

You can install Forge using same method as SD-WebUI. (Install Git, Python, Git Clone this repo and then run webui-user.bat).

**Or you can just use this one-click installation package (with git and python included).**

[>>> Click Here to Download One-Click Package<<<](https://github.com/lllyasviel/stable-diffusion-webui-forge/releases/download/latest/webui-forge.7z)

After you download, you uncompress, use `update.bat` to update, and use `run.bat` to run.

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/1251ee5c-33dc-4adc-8dd3-2b43af5ad425)


# Screenshots of Comparison

I tested with several devices, and this is a typical result from 8GB VRAM (3070ti laptop) with SDXL.

**This is original WebUI:**

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

You can see that Forge does not change WebUI results. Installing Forge is not a seed breaking change. 

Forge can perfectly keep WebUI unchanged even for most complicated prompts like `fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]`.

All your previous works still work in Forge!

Also, Forge promise that we will only do our jobs. We will not add unnecessary opinioned changes to UI. You are still using 100% Automatic1111 WebUI.

# Forge Backend

Forge backend removes all WebUI's codes related to resource management and reworked everything. All previous CMD flags like `medvram, lowvram, medvram-sdxl, precision full, no half, no half vae, attention_xxx, upcast unet`, ... are all **REMOVED**. Adding these flags will not cause error but they will not do anything now. **We highly encourage Forge users to remove all cmd flags and let Forge to decide how to load models.**

Without any cmd flag, Forge can run SDXL with 4GB vram and SD1.5 with 2GB vram.

**The only one flag that you may still need** is `--always-offload-from-vram` (This flag will make things **slower**). This option will let Forge always unload models from VRAM. This can be useful if you use multiple software together and want Forge to use less VRAM and give some vram to other software, or when you are using some old extensions that will compete vram with Forge, or (very rarely) when you get OOM.

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

# UNet Patcher

Now developing an extension is super simple. We finally have a patchable UNet.

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

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.
        
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

ControlNets can finally be called by different extensions.

Implementing Stable Video Diffusion and Zero123 are also super simple now (see also the codes). 

*Stable Video Diffusion:*

`extensions-builtin/sd_forge_svd/scripts/forge_svd.py`

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

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/d1a4a17d-f382-442d-91f2-fc5b6c10737f)

### Write a simple ControlNet:

Below is a simple extension to have a completely independent pass of ControlNet that never conflicts any other extensions:

`extensions-builtin/sd_forge_controlnet_example/scripts/sd_forge_controlnet_example.py`

Note that this extension is hidden because it is only for developers. To see it in UI, use `--show-controlnet-example`.

The memory optimization in this example is fully automatic. You do not need to care about memory and inference speed, but you may want to cache objects if you wish.

```python
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

```

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/0a703d8b-27df-4608-8b12-aff750f20ffa)


### Add a preprocessor

Below is the full codes to add a normalbae preprocessor with perfect memory managements.

You can use arbitrary independent extensions to add a preprocessor.

Your preprocessor will be read by all other extensions using `modules_forge.shared.preprocessors`

Below codes are in `extensions-builtin\forge_preprocessor_normalbae\scripts\preprocessor_normalbae.py`

```python
from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import preprocessor_dir, add_supported_preprocessor
from modules_forge.forge_util import resize_image_with_pad
from modules.modelloader import load_file_from_url

import types
import torch
import numpy as np

from einops import rearrange
from annotator.normalbae.models.NNET import NNET
from annotator.normalbae import load_checkpoint
from torchvision import transforms


class PreprocessorNormalBae(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'normalbae'
        self.tags = ['NormalMap']
        self.model_filename_filters = ['normal']
        self.slider_resolution = PreprocessorParameter(
            label='Resolution', minimum=128, maximum=2048, value=512, step=8, visible=True)
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.slider_3 = PreprocessorParameter(visible=False)
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 100  # higher goes to top in the list

    def load_model(self):
        if self.model_patcher is not None:
            return

        model_path = load_file_from_url(
            "https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt",
            model_dir=preprocessor_dir)

        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = load_checkpoint(model_path, model)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.model_patcher = self.setup_model_patcher(model)

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        input_image, remove_pad = resize_image_with_pad(input_image, resolution)

        self.load_model()

        self.move_all_model_patchers_to_gpu()

        assert input_image.ndim == 3
        image_normal = input_image

        with torch.no_grad():
            image_normal = self.send_tensor_to_model_device(torch.from_numpy(image_normal))
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            normal = self.model_patcher.model(image_normal)
            normal = normal[0][-1][:, :3]
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        return remove_pad(normal_image)


add_supported_preprocessor(PreprocessorNormalBae())

```

# New features (that are not available in original WebUI)

Thanks to Unet Patcher, many new things are possible now and supported in Forge, including SVD, Z123, masked Ip-adapter, masked controlnet, photomaker, etc.

Masked Ip-Adapter

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/75432c4b-4f2a-4027-b6ad-fcf48fffad2c)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/c8dad060-9f93-4781-a14d-255180787592)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/2680504e-6c7a-4531-b477-06e4c7b3ef3f)


Masked ControlNet

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/1d74e970-fbe3-40df-9c87-70c8426ec6e1)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/0bc7bc7f-1a33-41c6-89ea-f5b5e60f6c1e)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/10b07d97-a2b8-463c-928a-5ad9b78da25e)

PhotoMaker

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/377699d3-e6b5-4ace-9d0f-054ac1749fc8)


Marigold Depth

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/95787c05-3fe7-43cc-ba12-baf378284007)

# New Samplers (that are not in origin)

    DDPM
    DDPM Karras
    DPM++ 2M Turbo
    DPM++ 2M SDE Turbo
    LCM Karras
    Euler A Turbo

# About Extensions

ControlNet and TiledVAE are integrated, and you should uninstall these two extensions:

    sd-webui-controlnet
    multidiffusion-upscaler-for-automatic1111

Other extensions should work without problems, like:

    canvas-zoom
    translations/localizations
    Dynamic Prompts
    Adetailer
    Ultimate SD Upscale
    Reactor

However, if newer extensions use Forge, their codes can be much shorter. 

Usually if an old extension rework using Forge's unet patcher, 80% codes can be removed, especially when they need to call controlnet.

Note that **animatediff** is under construction by [continue-revolution](https://github.com/continue-revolution) and coming soon. (continue-revolution original words: "basic features (t2v, prompt travel, inf v2v) have been proven to work well, motion lora, cn v2v still under construction and may be finished in a week, and we can mention motion brush")
