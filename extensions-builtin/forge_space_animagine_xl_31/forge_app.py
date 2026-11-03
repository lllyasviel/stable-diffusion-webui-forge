import spaces
import os
import gc
import gradio as gr
import numpy as np
import torch
import json
import config
import utils
import logging
from PIL import Image, PngImagePlugin
from datetime import datetime
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DESCRIPTION = "Animagine XL 3.1"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU. </p>"
IS_COLAB = utils.is_google_colab() or os.getenv("IS_COLAB") == "1"
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MIN_IMAGE_SIZE = int(os.getenv("MIN_IMAGE_SIZE", "512"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

MODEL = os.getenv(
    "MODEL",
    "https://huggingface.co/cagliostrolab/animagine-xl-3.1/blob/main/animagine-xl-3.1.safetensors",
)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pipeline(model_name):
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    pipeline = (
        StableDiffusionXLPipeline.from_single_file
        if MODEL.endswith(".safetensors")
        else StableDiffusionXLPipeline.from_pretrained
    )

    pipe = pipeline(
        model_name,
        vae=vae,
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",
        use_safetensors=True,
        add_watermarker=False,
        use_auth_token=HF_TOKEN,
    )

    # pipe.to(device)
    return pipe


with spaces.capture_gpu_object() as gpu_object:
    pipe = load_pipeline(MODEL)
    logger.info("Loaded on Device!")


spaces.automatically_move_pipeline_components(pipe)
spaces.change_attention_from_diffusers_to_forge(pipe.unet)
spaces.change_attention_from_diffusers_to_forge(pipe.vae)


@spaces.GPU(gpu_objects=[gpu_object], manual_load=True)
def generate(
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
        custom_width: int = 1024,
        custom_height: int = 1024,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 28,
        sampler: str = "Euler a",
        aspect_ratio_selector: str = "896 x 1152",
        style_selector: str = "(None)",
        quality_selector: str = "Standard v3.1",
        use_upscaler: bool = False,
        upscaler_strength: float = 0.55,
        upscale_by: float = 1.5,
        add_quality_tags: bool = True,
        progress=gr.Progress(track_tqdm=True),
):
    generator = utils.seed_everything(seed)

    width, height = utils.aspect_ratio_handler(
        aspect_ratio_selector,
        custom_width,
        custom_height,
    )

    prompt = utils.add_wildcard(prompt, wildcard_files)

    prompt, negative_prompt = utils.preprocess_prompt(
        quality_prompt, quality_selector, prompt, negative_prompt, add_quality_tags
    )
    prompt, negative_prompt = utils.preprocess_prompt(
        styles, style_selector, prompt, negative_prompt
    )

    width, height = utils.preprocess_image_dimensions(width, height)

    backup_scheduler = pipe.scheduler
    pipe.scheduler = utils.get_scheduler(pipe.scheduler.config, sampler)

    if use_upscaler:
        upscaler_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": f"{width} x {height}",
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "sampler": sampler,
        "sdxl_style": style_selector,
        "add_quality_tags": add_quality_tags,
        "quality_tags": quality_selector,
    }

    if use_upscaler:
        new_width = int(width * upscale_by)
        new_height = int(height * upscale_by)
        metadata["use_upscaler"] = {
            "upscale_method": "nearest-exact",
            "upscaler_strength": upscaler_strength,
            "upscale_by": upscale_by,
            "new_resolution": f"{new_width} x {new_height}",
        }
    else:
        metadata["use_upscaler"] = None
        metadata["Model"] = {
            "Model": DESCRIPTION,
            "Model hash": "e3c47aedb0",
        }

    logger.info(json.dumps(metadata, indent=4))

    try:
        if use_upscaler:
            latents = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="latent",
            ).images
            upscaled_latents = utils.upscale(latents, "nearest-exact", upscale_by)
            images = upscaler_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=upscaled_latents,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=upscaler_strength,
                generator=generator,
                output_type="pil",
            ).images
        else:
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
            ).images

        if images:
            image_paths = [
                utils.save_image(image, metadata, OUTPUT_DIR, IS_COLAB)
                for image in images
            ]

            for image_path in image_paths:
                logger.info(f"Image saved as {image_path} with metadata")

        return image_paths, metadata
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise
    finally:
        if use_upscaler:
            del upscaler_pipe
        pipe.scheduler = backup_scheduler
        utils.free_memory()



styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in config.style_list}
quality_prompt = {
    k["name"]: (k["prompt"], k["negative_prompt"]) for k in config.quality_prompt_list
}

wildcard_files = utils.load_wildcard_files(spaces.convert_root_path() + "wildcard")

with gr.Blocks(css="style.css", theme="NoCrypt/miku@1.2.1") as demo:
    title = gr.HTML(
        f"""<h1><span>{DESCRIPTION}</span></h1>""",
        elem_id="title",
    )
    gr.Markdown(
        f"""Gradio demo for [cagliostrolab/animagine-xl-3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1)""",
        elem_id="subtitle",
    )
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("Txt2img"):
                with gr.Group():
                    prompt = gr.Text(
                        label="Prompt",
                        max_lines=5,
                        placeholder="Enter your prompt",
                    )
                    negative_prompt = gr.Text(
                        label="Negative Prompt",
                        max_lines=5,
                        placeholder="Enter a negative prompt",
                    )
                    with gr.Accordion(label="Quality Tags", open=True):
                        add_quality_tags = gr.Checkbox(
                            label="Add Quality Tags", value=True
                        )
                        quality_selector = gr.Dropdown(
                            label="Quality Tags Presets",
                            interactive=True,
                            choices=list(quality_prompt.keys()),
                            value="Standard v3.1",
                        )
            with gr.Tab("Advanced Settings"):
                with gr.Group():
                    style_selector = gr.Radio(
                        label="Style Preset",
                        container=True,
                        interactive=True,
                        choices=list(styles.keys()),
                        value="(None)",
                    )
                with gr.Group():
                    aspect_ratio_selector = gr.Radio(
                        label="Aspect Ratio",
                        choices=config.aspect_ratios,
                        value="896 x 1152",
                        container=True,
                    )
                with gr.Group(visible=False) as custom_resolution:
                    with gr.Row():
                        custom_width = gr.Slider(
                            label="Width",
                            minimum=MIN_IMAGE_SIZE,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                        custom_height = gr.Slider(
                            label="Height",
                            minimum=MIN_IMAGE_SIZE,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                with gr.Group():
                    use_upscaler = gr.Checkbox(label="Use Upscaler", value=False)
                    with gr.Row() as upscaler_row:
                        upscaler_strength = gr.Slider(
                            label="Strength",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.55,
                            visible=False,
                        )
                        upscale_by = gr.Slider(
                            label="Upscale by",
                            minimum=1,
                            maximum=1.5,
                            step=0.1,
                            value=1.5,
                            visible=False,
                        )
                with gr.Group():
                    sampler = gr.Dropdown(
                        label="Sampler",
                        choices=config.sampler_list,
                        interactive=True,
                        value="Euler a",
                    )
                with gr.Group():
                    seed = gr.Slider(
                        label="Seed", minimum=0, maximum=utils.MAX_SEED, step=1, value=0
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Group():
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=1,
                            maximum=12,
                            step=0.1,
                            value=7.0,
                        )
                        num_inference_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=28,
                        )
        with gr.Column(scale=3):
            with gr.Blocks():
                run_button = gr.Button("Generate", variant="primary")
            result = gr.Gallery(
                label="Result",
                columns=1,
                height='100%',
                preview=True,
                show_label=False
            )
            with gr.Accordion(label="Generation Parameters", open=False):
                gr_metadata = gr.JSON(label="metadata", show_label=False)
            gr.Examples(
                examples=config.examples,
                inputs=prompt,
                outputs=[result, gr_metadata],
                fn=lambda *args, **kwargs: generate(*args, use_upscaler=True, **kwargs),
                cache_examples=CACHE_EXAMPLES,
            )
    use_upscaler.change(
        fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
        inputs=use_upscaler,
        outputs=[upscaler_strength, upscale_by],
        queue=False,
        api_name=False,
    )
    aspect_ratio_selector.change(
        fn=lambda x: gr.update(visible=x == "Custom"),
        inputs=aspect_ratio_selector,
        outputs=custom_resolution,
        queue=False,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=utils.randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            custom_width,
            custom_height,
            guidance_scale,
            num_inference_steps,
            sampler,
            aspect_ratio_selector,
            style_selector,
            quality_selector,
            use_upscaler,
            upscaler_strength,
            upscale_by,
            add_quality_tags,
        ],
        outputs=[result, gr_metadata],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(debug=IS_COLAB, share=IS_COLAB)
