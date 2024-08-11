# Under Construction

WebUI Forge is under a week of major revision right now between 2024 Aug 1 and Aug 10. To join the test, just update to the latest unstable version.

**Current Progress (2024 Aug 10):** Backend Rewrite is 95% finished.

Update Aug 11: Hey everyone it seems that xformers are somewhat broken now - if you see "NoneType object is not iterable", just uninstall xformers. I will find a fix later when I get more free time.

# Stable Diffusion WebUI Forge

Stable Diffusion WebUI Forge is a platform on top of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) (based on [Gradio](https://www.gradio.app/)) to make development easier, optimize resource management, speed up inference, and study experimental features.

The name "Forge" is inspired from "Minecraft Forge". This project is aimed at becoming SD WebUI's Forge.

Forge is currently based on SD-WebUI 1.10.1 at [this commit](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/82a973c04367123ae98bd9abdf80d9eda9b910e2).

# Installing Forge

If you are proficient in Git and you want to install Forge as another branch of SD-WebUI, please see [here](https://github.com/continue-revolution/sd-webui-animatediff/blob/forge/master/docs/how-to-use.md#you-have-a1111-and-you-know-git). In this way, you can reuse all SD checkpoints and all extensions you installed previously in your OG SD-WebUI, but you should know what you are doing.

If you know what you are doing, you can install Forge using same method as SD-WebUI. (Install Git, Python, Git Clone the forge repo `https://github.com/lllyasviel/stable-diffusion-webui-forge.git` and then run webui-user.bat).

**Or you can just use this one-click installation package (with git and python included).**

[>>> Click Here to Download One-Click Package (CUDA 12.4) <<<](https://github.com/lllyasviel/stable-diffusion-webui-forge/releases/download/latest/webui_forge_cu124_torch24.7z)

If your device does not support CUDA 12.4 (if you get error "Torch is not able to use GPU"), here is [the CUDA 12.1 version](https://github.com/lllyasviel/stable-diffusion-webui-forge/releases/download/latest/webui_forge_cu121_torch21.7z).

After you download, you uncompress, use `update.bat` to update, and use `run.bat` to run.

Note that running `update.bat` is important, otherwise you may be using a previous version with potential bugs unfixed.

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/c49bd60d-82bd-4086-9859-88d472582b94)

### Previous Versions

You can download previous versions [here](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/849).

# Forge Status

Based on manual test one-by-one:

| Component                                         | Status  | Last Test    |
|---------------------------------------------------|---------|--------------|
| Basic Diffusion                                   | Normal  | 2024 July 27 |
| GPU Memory Management System                      | Normal  | 2024 July 27 |
| LoRAs                                             | Normal  | 2024 July 27 |
| All Preprocessors                                 | Normal  | 2024 July 27 |
| All ControlNets                                   | Normal  | 2024 July 27 |
| All IP-Adapters                                   | Normal  | 2024 July 27 |
| All Instant-IDs                                   | Normal  | 2024 July 27 |
| All Reference-only Methods                        | Normal  | 2024 July 27 |
| All Integrated Extensions                         | Normal  | 2024 July 27 |
| Popular Extensions (Adetailer, etc)               | Normal  | 2024 July 27 |
| Gradio 4 UIs                                      | Normal  | 2024 July 27 |
| Gradio 4 Forge Canvas                             | Normal  | 2024 July 27 |
| LoRA/Checkpoint Selection UI for Gradio 4         | Normal  | 2024 July 27 |
| Photopea/OpenposeEditor/etc for ControlNet        | Normal  | 2024 July 27 |
| Wacom 128 level touch pressure support for Canvas | Normal  | 2024 July 15 |
| Microsoft Surface touch pressure support for Canvas | Broken, pending fix  | 2024 July 29 |

Feel free to open issue if anything is broken and I will take a look every several days. If I do not update this "Forge Status" then it means I cannot reproduce any problem. In that case, fresh re-install should help most.

# Under Construction

This Readme is under construction ... more docs/wiki coming soon ...
