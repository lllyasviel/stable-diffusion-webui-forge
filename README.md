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

I tested with several devices, and this is a typical result from 8GB VRAM (3070ti laptop).

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



