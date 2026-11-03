import spaces
import torch
import gradio as gr
from gradio import processing_utils, utils
from PIL import Image
import random

from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
import tempfile
import time
from share_btn import community_icon_html, loading_icon_html, share_js
import user_history
from illusion_style import css
import os
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"


with spaces.capture_gpu_object() as gpu_object:
    # Initialize both pipelines
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)

    # Initialize the safety checker conditionally
    SAFETY_CHECKER_ENABLED = os.environ.get("SAFETY_CHECKER", "0") == "1"
    safety_checker = None
    feature_extractor = None
    if SAFETY_CHECKER_ENABLED:
        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        vae=vae,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        torch_dtype=torch.float16,
    )

cn_and_unet = torch.nn.ModuleList([main_pipe.unet, main_pipe.controlnet])

spaces.automatically_move_to_gpu_when_forward(main_pipe.unet, target_model=cn_and_unet)
spaces.automatically_move_to_gpu_when_forward(main_pipe.controlnet, target_model=cn_and_unet)
spaces.automatically_move_to_gpu_when_forward(main_pipe.vae)
spaces.automatically_move_to_gpu_when_forward(main_pipe.text_encoder)
spaces.change_attention_from_diffusers_to_forge(main_pipe.unet)
spaces.change_attention_from_diffusers_to_forge(main_pipe.vae)

# Function to check NSFW images
# def check_nsfw_images(images: list[Image.Image]) -> tuple[list[Image.Image], list[bool]]:
#    if SAFETY_CHECKER_ENABLED:
#        safety_checker_input = feature_extractor(images, return_tensors="pt").to("cuda")
#        has_nsfw_concepts = safety_checker(
#            images=[images],
#            clip_input=safety_checker_input.pixel_values.to("cuda")
#        )
#        return images, has_nsfw_concepts
#    else:
#        return images, [False] * len(images)

# main_pipe.unet = torch.compile(main_pipe.unet, mode="reduce-overhead", fullgraph=True)
# main_pipe.unet.to(memory_format=torch.channels_last)
# main_pipe.unet = torch.compile(main_pipe.unet, mode="reduce-overhead", fullgraph=True)
# model_id = "stabilityai/sd-x2-latent-upscaler"
image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)

# image_pipe.unet = torch.compile(image_pipe.unet, mode="reduce-overhead", fullgraph=True)
# upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# upscaler.to("cuda")


# Sampler map
SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}


def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)
    left = (width - new_dimension) / 2
    top = (height - new_dimension) / 2
    right = (width + new_dimension) / 2
    bottom = (height + new_dimension) / 2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img


def common_upscale(samples, width, height, upscale_method, crop=False):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:, :, y:old_height - y, x:old_width - x]
    else:
        s = samples

    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)


def upscale(samples, upscale_method, scale_by):
    # s = samples.copy()
    width = round(samples["images"].shape[3] * scale_by)
    height = round(samples["images"].shape[2] * scale_by)
    s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
    return (s)


def check_inputs(prompt: str, control_image: Image.Image):
    if control_image is None:
        raise gr.Error("Please select or upload an Input Illusion")
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")


def convert_to_pil(base64_image):
    pil_image = Image.open(base64_image)
    return pil_image


def convert_to_base64(pil_image):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        image.save(temp_file.name)
    return temp_file.name


# Inference function
@spaces.GPU(gpu_objects=[gpu_object], manual_load=True)
def inference(
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float = 8.0,
        controlnet_conditioning_scale: float = 1,
        control_guidance_start: float = 1,
        control_guidance_end: float = 1,
        upscaler_strength: float = 0.5,
        seed: int = -1,
        sampler="DPM++ Karras SDE",
        progress=gr.Progress(track_tqdm=True),
        profile=None,
):
    start_time = time.time()
    start_time_struct = time.localtime(start_time)
    start_time_formatted = time.strftime("%H:%M:%S", start_time_struct)
    print(f"Inference started at {start_time_formatted}")

    # Generate the initial image
    # init_image = init_pipe(prompt).images[0]

    # Rest of your existing code
    control_image_small = center_crop_resize(control_image)
    control_image_large = center_crop_resize(control_image, (1024, 1024))

    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)
    my_seed = random.randint(0, 2 ** 32 - 1) if seed == -1 else seed
    generator = torch.Generator(device="cuda").manual_seed(my_seed)

    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        num_inference_steps=15,
        output_type="latent"
    )
    upscaled_latents = upscale(out, "nearest-exact", 2)
    out_image = image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image_large,
        image=upscaled_latents,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=20,
        strength=upscaler_strength,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale)
    )
    end_time = time.time()
    end_time_struct = time.localtime(end_time)
    end_time_formatted = time.strftime("%H:%M:%S", end_time_struct)
    print(f"Inference ended at {end_time_formatted}, taking {end_time - start_time}s")

    # Save image + metadata
    user_history.save_image(
        label=prompt,
        image=out_image["images"][0],
        profile=profile,
        metadata={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "upscaler_strength": upscaler_strength,
            "seed": seed,
            "sampler": sampler,
        },
    )

    return out_image["images"][0], gr.update(visible=True), gr.update(visible=True), my_seed


with gr.Blocks() as app:
    gr.Markdown(
        '''
        <div style="text-align: center;">
            <h1>Illusion Diffusion HQ 🌀</h1>
            <p style="font-size:16px;">Generate stunning high quality illusion artwork with Stable Diffusion</p>
            <p>Illusion Diffusion is back up with a safety checker! Because I have been asked, if you would like to support me, consider using <a href="https://deforum.studio">deforum.studio</a></p>
            <p>A space by AP <a href="https://twitter.com/angrypenguinPNG">Follow me on Twitter</a> with big contributions from <a href="https://twitter.com/multimodalart">multimodalart</a></p>
            <p>This project works by using <a href="https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster">Monster Labs QR Control Net</a>. Given a prompt and your pattern, we use a QR code conditioned controlnet to create a stunning illusion! Credit to: <a href="https://twitter.com/MrUgleh">MrUgleh</a> for discovering the workflow :)</p>
        </div>
        '''
    )

    state_img_input = gr.State()
    state_img_output = gr.State()
    with gr.Row():
        with gr.Column():
            control_image = gr.Image(label="Input Illusion", type="pil", elem_id="control_image")
            controlnet_conditioning_scale = gr.Slider(minimum=0.0, maximum=5.0, step=0.01, value=0.8, label="Illusion strength", elem_id="illusion_strength", info="ControlNet conditioning scale")
            examples = ["checkers.png", "checkers_mid.jpg", "pattern.png", "ultra_checkers.png", "spiral.jpeg", "funky.jpeg"]
            examples = [spaces.convert_root_path() + y for y in examples]
            gr.Examples(examples=examples, inputs=control_image)
            prompt = gr.Textbox(label="Prompt", elem_id="prompt", info="Type what you want to generate", placeholder="Medieval village scene with busy streets and castle in the distance")
            negative_prompt = gr.Textbox(label="Negative Prompt", info="Type what you don't want to see", value="low quality", elem_id="negative_prompt")
            with gr.Accordion(label="Advanced Options", open=False):
                guidance_scale = gr.Slider(minimum=0.0, maximum=50.0, step=0.25, value=7.5, label="Guidance Scale")
                sampler = gr.Dropdown(choices=list(SAMPLER_MAP.keys()), value="Euler")
                control_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0, label="Start of ControlNet")
                control_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=1, label="End of ControlNet")
                strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=1, label="Strength of the upscaler")
                seed = gr.Slider(minimum=-1, maximum=9999999999, step=1, value=-1, label="Seed", info="-1 means random seed")
                used_seed = gr.Number(label="Last seed used", interactive=False)
            run_btn = gr.Button("Run")
        with gr.Column():
            result_image = gr.Image(label="Illusion Diffusion Output", interactive=False, elem_id="output")
            with gr.Group(elem_id="share-btn-container", visible=False) as share_group:
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")

    prompt.submit(
        check_inputs,
        inputs=[prompt, control_image],
        queue=False
    ).success(
        inference,
        inputs=[control_image, prompt, negative_prompt, guidance_scale, controlnet_conditioning_scale, control_start, control_end, strength, seed, sampler],
        outputs=[result_image, result_image, share_group, used_seed])

    run_btn.click(
        check_inputs,
        inputs=[prompt, control_image],
        queue=False
    ).success(
        inference,
        inputs=[control_image, prompt, negative_prompt, guidance_scale, controlnet_conditioning_scale, control_start, control_end, strength, seed, sampler],
        outputs=[result_image, result_image, share_group, used_seed])

    share_button.click(None, [], [], js=share_js)

with gr.Blocks(css=css) as demo:
    with gr.Tab("Demo"):
        app.render()
    with gr.Tab("Past generations"):
        user_history.render()

if __name__ == "__main__":
    demo.launch()
