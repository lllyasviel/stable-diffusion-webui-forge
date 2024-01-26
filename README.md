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
