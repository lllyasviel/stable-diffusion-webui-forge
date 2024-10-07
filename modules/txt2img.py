import json
from contextlib import closing

import modules.scripts
from modules import processing, infotext_utils
from modules.infotext_utils import create_override_settings_dict, parse_generation_parameters
from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html
from PIL import Image
import gradio as gr
from modules_forge import main_thread

# modules/txt2img.py

from modules.shared import state
from modules.processing import Processed
import os

# modules/txt2img.py

from modules.shared import state
from modules.processing import Processed
import os

import json
import re
import itertools
def txt2img_multi_prompt(id_task: str, request: gr.Request, *args, **kwargs):
    # Convert args to a list so we can modify it
    args = list(args)

    # Adjusted index mapping based on debug output
    prompt_index = 0          # args[0] - prompt
    batch_count_index = 3     # args[3] - batch_count
    batch_size_index = 4      # args[4] - batch_size

    # Extract the prompt from args
    prompt = args[prompt_index]
    print(f"DEBUG: prompt = {prompt}")

    if not prompt:
        return [], '', '', ''  # Return empty outputs if no prompt provided

    # Split the prompt into multiple sub-prompts
    sub_prompts = [p.strip() for p in prompt.split('\n') if p.strip()]
    if not sub_prompts:
        return [], '', '', ''  # Return empty outputs if no valid prompts provided

    # ** New Code Start **

    # Function to expand a single prompt with alternating tags
    def expand_prompt(prompt):
        # Regular expression to find [option1|option2|...]
        pattern = re.compile(r'\[([^\[\]]+)\]')
        matches = pattern.findall(prompt)

        # If no matches, return the prompt as-is
        if not matches:
            return [prompt]

        # List to hold all options lists
        options_list = []

        # For each match, split the options
        for match in matches:
            options = match.split('|')
            options_list.append(options)

        # Generate all combinations using itertools.product
        combinations = list(itertools.product(*options_list))

        # Generate expanded prompts by replacing the placeholders with combinations
        expanded_prompts = []
        for combo in combinations:
            temp_prompt = prompt
            for match, replacement in zip(matches, combo):
                # Replace the first occurrence of the matched pattern
                temp_prompt = pattern.sub(replacement, temp_prompt, count=1)
            expanded_prompts.append(temp_prompt)

        return expanded_prompts

    # Expand each sub-prompt
    expanded_prompts = []
    for sub_prompt in sub_prompts:
        expanded_prompts.extend(expand_prompt(sub_prompt))

    # ** New Code End **

    # Get batch_count and batch_size for progress tracking
    batch_count = args[batch_count_index]
    batch_size = args[batch_size_index]

    # Update job count for progress tracking
    total_prompts = len(expanded_prompts)
    state.job_count = total_prompts * batch_count * batch_size

    # Initialize collections to gather outputs
    all_images = []
    all_info_dicts = []
    all_infotexts = []
    all_comments = []

    for i, expanded_prompt in enumerate(expanded_prompts):
        # Update the prompt in args
        args[prompt_index] = expanded_prompt  # Update 'prompt' in args

        # Update the job state for progress tracking
        state.job = f"Prompt {i + 1}/{total_prompts}"

        # Call the original txt2img function with updated args
        processed = txt2img(id_task, request, *args, **kwargs)

        # Unpack the processed tuple
        images, info_json, info_html, comments_html = processed

        # Collect images
        all_images.extend(images)

        # Parse info_json and collect info dictionaries
        try:
            info_dict = json.loads(info_json)
            all_info_dicts.append(info_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing info_json: {e}")

        # Collect infotexts and comments
        all_infotexts.append(info_html)
        all_comments.append(comments_html)

        # Check for interruption
        if state.interrupted:
            break

    # Aggregate info_json by combining all info dictionaries into a list
    combined_info_json = json.dumps(all_info_dicts)

    # Combine infotexts and comments into single strings
    combined_info_html = '\n'.join(all_infotexts)
    combined_comments_html = '\n'.join(all_comments)

    # Return the aggregated outputs
    return all_images, combined_info_json, combined_info_html, combined_comments_html

def txt2img_create_processing(id_task: str, request: gr.Request, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt, hr_cfg: float, hr_distilled_cfg: float, override_settings_texts, *args, force_enable_hr=False):
    override_settings = create_override_settings_dict(override_settings_texts)

    if force_enable_hr:
        enable_hr = True

    p = processing.StableDiffusionProcessingTxt2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        n_iter=n_iter,
        cfg_scale=cfg_scale,
        distilled_cfg_scale=distilled_cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_scheduler=None if hr_scheduler == 'Use same scheduler' else hr_scheduler,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        hr_cfg=hr_cfg,
        hr_distilled_cfg=hr_distilled_cfg,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    return p


def txt2img_upscale_function(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    assert len(gallery) > 0, 'No image to upscale'

    if gallery_index < 0 or gallery_index >= len(gallery):
        return gallery, generation_info, f'Bad image index: {gallery_index}', ''

    geninfo = json.loads(generation_info)

    #   catch situation where user tries to hires-fix the grid: probably a mistake, results can be bad aspect ratio - just don't do it
    first_image_index = geninfo.get('index_of_first_image', 0)
    #   catch if user tries to upscale a control image, this function will fail later trying to get infotext that doesn't exist
    count_images = len(geninfo.get('infotexts'))        #   note: we have batch_size in geninfo, but not batch_count
    if len(gallery) > 1 and (gallery_index < first_image_index or gallery_index >= count_images):
        return gallery, generation_info, 'Unable to upscale grid or control images.', ''

    p = txt2img_create_processing(id_task, request, *args, force_enable_hr=True)
    p.batch_size = 1
    p.n_iter = 1
    # txt2img_upscale attribute that signifies this is called by txt2img_upscale
    p.txt2img_upscale = True

    image_info = gallery[gallery_index]
    p.firstpass_image = infotext_utils.image_from_url_text(image_info)

    parameters = parse_generation_parameters(geninfo.get('infotexts')[gallery_index], [])
    p.seed = parameters.get('Seed', -1)
    p.subseed = parameters.get('Variation seed', -1)

    #   update processing width/height based on actual dimensions of source image
    p.width = gallery[gallery_index][0].size[0]
    p.height = gallery[gallery_index][0].size[1]
    p.extra_generation_params['Original Size'] = f'{args[8]}x{args[7]}'

    p.override_settings['save_images_before_highres_fix'] = False

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    new_gallery = []
    for i, image in enumerate(gallery):
        if i == gallery_index:
            new_gallery.extend(processed.images)
        else:
            new_gallery.append(image)

    geninfo["infotexts"][gallery_index] = processed.info

    return new_gallery, json.dumps(geninfo), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def txt2img_function(id_task: str, request: gr.Request, *args):
    p = txt2img_create_processing(id_task, request, *args)

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images + processed.extra_images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def txt2img_upscale(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    return main_thread.run_and_wait_result(txt2img_upscale_function, id_task, request, gallery, gallery_index, generation_info, *args)


def txt2img(id_task: str, request: gr.Request, *args):
    return main_thread.run_and_wait_result(txt2img_function, id_task, request, *args)
