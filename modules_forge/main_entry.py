import os
import torch
import gradio as gr

from gradio.context import Context
from modules import shared_items, shared, ui_common, sd_models, processing, infotext_utils, paths
from backend import memory_management, stream


total_vram = int(memory_management.total_vram)

ui_forge_preset: gr.Radio = None

ui_checkpoint: gr.Dropdown = None
ui_vae: gr.Dropdown = None
ui_clip_skip: gr.Slider = None

ui_forge_unet_storage_dtype_options: gr.Radio = None
ui_forge_async_loading: gr.Radio = None
ui_forge_pin_shared_memory: gr.Radio = None
ui_forge_inference_memory: gr.Slider = None

forge_unet_storage_dtype_options = {
    'Automatic': None,
    'bnb-nf4': 'nf4',
    'float8-e4m3fn': torch.float8_e4m3fn,
    'bnb-fp4': 'fp4',
    'float8-e5m2': torch.float8_e5m2,
}

module_list = {}


def bind_to_opts(comp, k, save=False, callback=None):
    def on_change(v):
        shared.opts.set(k, v)
        if save:
            shared.opts.save(shared.config_filename)
        if callback is not None:
            callback()
        return

    comp.change(on_change, inputs=[comp], queue=False, show_progress=False)
    return


def make_checkpoint_manager_ui():
    global ui_checkpoint, ui_vae, ui_clip_skip, ui_forge_unet_storage_dtype_options, ui_forge_async_loading, ui_forge_pin_shared_memory, ui_forge_inference_memory, ui_forge_preset

    if shared.opts.sd_model_checkpoint in [None, 'None', 'none', '']:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set('sd_model_checkpoint', next(iter(sd_models.checkpoints_list.values())).name)

    ui_forge_preset = gr.Radio(label="UI", value=lambda: shared.opts.forge_preset, choices=['sd', 'xl', 'flux', 'all'])

    ckpt_list, vae_list = refresh_models()

    ui_checkpoint = gr.Dropdown(
        value=lambda: shared.opts.sd_model_checkpoint,
        label="Checkpoint",
        elem_classes=['model_selection'],
        choices=ckpt_list
    )

    ui_vae = gr.Dropdown(
        value=lambda: [os.path.basename(x) for x in shared.opts.forge_additional_modules],
        multiselect=True,
        label="VAE / Text Encoder",
        render=False,
        choices=vae_list
    )

    def gr_refresh_models():
        a, b = refresh_models()
        return gr.update(choices=a), gr.update(choices=b)

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id=f"forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(
        fn=gr_refresh_models,
        inputs=[],
        outputs=[ui_checkpoint, ui_vae],
        show_progress=False,
        queue=False
    )
    Context.root_block.load(
        fn=gr_refresh_models,
        inputs=[],
        outputs=[ui_checkpoint, ui_vae],
        show_progress=False,
        queue=False
    )

    ui_vae.render()

    ui_forge_unet_storage_dtype_options = gr.Dropdown(label="Diffusion in Low Bits", value=lambda: shared.opts.forge_unet_storage_dtype, choices=list(forge_unet_storage_dtype_options.keys()))
    bind_to_opts(ui_forge_unet_storage_dtype_options, 'forge_unet_storage_dtype', save=True, callback=refresh_model_loading_parameters)

    ui_forge_async_loading = gr.Radio(label="Swap Method", value=lambda: shared.opts.forge_async_loading, choices=['Queue', 'Async'])
    ui_forge_pin_shared_memory = gr.Radio(label="Swap Location", value=lambda: shared.opts.forge_pin_shared_memory, choices=['CPU', 'Shared'])
    ui_forge_inference_memory = gr.Slider(label="GPU Weights (MB)", value=lambda: total_vram - shared.opts.forge_inference_memory, minimum=0, maximum=int(memory_management.total_vram), step=1)

    mem_comps = [ui_forge_inference_memory, ui_forge_async_loading, ui_forge_pin_shared_memory]

    ui_forge_inference_memory.change(refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_async_loading.change(refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_pin_shared_memory.change(refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    Context.root_block.load(refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    ui_clip_skip = gr.Slider(label="Clip skip", value=lambda: shared.opts.CLIP_stop_at_last_layers, **{"minimum": 1, "maximum": 12, "step": 1})
    bind_to_opts(ui_clip_skip, 'CLIP_stop_at_last_layers', save=False)

    ui_checkpoint.change(checkpoint_change, inputs=[ui_checkpoint], show_progress=False)
    ui_vae.change(vae_change, inputs=[ui_vae], queue=False, show_progress=False)

    return


def find_files_with_extensions(base_path, extensions):
    found_files = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                found_files[file] = full_path
    return found_files


def refresh_models():
    global module_list

    shared_items.refresh_checkpoints()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)

    file_extensions = ['ckpt', 'pt', 'bin', 'safetensors']

    module_list.clear()
    
    module_paths = [
        os.path.abspath(os.path.join(paths.models_path, "VAE")),
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
    ]

    if isinstance(shared.cmd_opts.vae_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.vae_dir))

    for vae_path in module_paths:
        vae_files = find_files_with_extensions(vae_path, file_extensions)
        module_list.update(vae_files)

    return ckpt_list, module_list.keys()


def refresh_memory_management_settings(model_memory, async_loading, pin_shared_memory):
    inference_memory = total_vram - model_memory

    shared.opts.set('forge_async_loading', async_loading)
    shared.opts.set('forge_inference_memory', inference_memory)
    shared.opts.set('forge_pin_shared_memory', pin_shared_memory)

    stream.stream_activated = async_loading == 'Async'
    memory_management.current_inference_memory = inference_memory * 1024 * 1024
    memory_management.PIN_SHARED_MEMORY = pin_shared_memory == 'Shared'

    log_dict = dict(
        stream=stream.should_use_stream(),
        inference_memory=memory_management.minimum_inference_memory() / (1024 * 1024),
        pin_shared_memory=memory_management.PIN_SHARED_MEMORY
    )

    print(f'Environment vars changed: {log_dict}')

    processing.need_global_unload = True
    return


def refresh_model_loading_parameters():
    from modules.sd_models import select_checkpoint, model_data

    checkpoint_info = select_checkpoint()

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=shared.opts.forge_additional_modules,
        unet_storage_dtype=forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, None)
    )

    print(f'Model selected: {model_data.forge_loading_parameters}')

    return


def checkpoint_change(ckpt_name):
    shared.opts.set('sd_model_checkpoint', ckpt_name)
    shared.opts.save(shared.config_filename)

    refresh_model_loading_parameters()
    return


def vae_change(module_names):
    modules = []

    for n in module_names:
        if n in module_list:
            modules.append(module_list[n])

    shared.opts.set('forge_additional_modules', modules)
    shared.opts.save(shared.config_filename)

    refresh_model_loading_parameters()
    return


def get_a1111_ui_component(tab, label):
    fields = infotext_utils.paste_fields[tab]['fields']
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_width = get_a1111_ui_component('txt2img', 'Size-1')
    ui_txt2img_height = get_a1111_ui_component('txt2img', 'Size-2')
    ui_txt2img_cfg = get_a1111_ui_component('txt2img', 'CFG scale')
    ui_txt2img_distilled_cfg = get_a1111_ui_component('txt2img', 'Distilled CFG Scale')
    ui_txt2img_sampler = get_a1111_ui_component('txt2img', 'sampler_name')
    ui_txt2img_scheduler = get_a1111_ui_component('txt2img', 'scheduler')

    ui_img2img_width = get_a1111_ui_component('img2img', 'Size-1')
    ui_img2img_height = get_a1111_ui_component('img2img', 'Size-2')
    ui_img2img_cfg = get_a1111_ui_component('img2img', 'CFG scale')
    ui_img2img_distilled_cfg = get_a1111_ui_component('img2img', 'Distilled CFG Scale')
    ui_img2img_sampler = get_a1111_ui_component('img2img', 'sampler_name')
    ui_img2img_scheduler = get_a1111_ui_component('img2img', 'scheduler')

    output_targets = [
        ui_vae,
        ui_clip_skip,
        ui_forge_unet_storage_dtype_options,
        ui_forge_async_loading,
        ui_forge_pin_shared_memory,
        ui_forge_inference_memory,
        ui_txt2img_width,
        ui_img2img_width,
        ui_txt2img_height,
        ui_img2img_height,
        ui_txt2img_cfg,
        ui_img2img_cfg,
        ui_txt2img_distilled_cfg,
        ui_img2img_distilled_cfg,
        ui_txt2img_sampler,
        ui_img2img_sampler,
        ui_txt2img_scheduler,
        ui_img2img_scheduler
    ]

    ui_forge_preset.change(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)
    Context.root_block.load(on_preset_change, inputs=None, outputs=output_targets, queue=False, show_progress=False)

    refresh_model_loading_parameters()
    return


def on_preset_change(preset=None):
    if preset is not None:
        shared.opts.set('forge_preset', preset)
        shared.opts.save(shared.config_filename)

    if shared.opts.forge_preset == 'sd':
        return [
            gr.update(visible=True),  # ui_vae
            gr.update(visible=True, value=1),  # ui_clip_skip
            gr.update(visible=False, value='Automatic'),  # ui_forge_unet_storage_dtype_options
            gr.update(visible=False, value='Queue'),  # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),  # ui_forge_pin_shared_memory
            gr.update(visible=False, value=total_vram - 1024),  # ui_forge_inference_memory
            gr.update(value=512),  # ui_txt2img_width
            gr.update(value=512),  # ui_img2img_width
            gr.update(value=640),  # ui_txt2img_height
            gr.update(value=512),  # ui_img2img_height
            gr.update(value=7),  # ui_txt2img_cfg
            gr.update(value=7),  # ui_img2img_cfg
            gr.update(visible=False, value=3.5),  # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),  # ui_img2img_distilled_cfg
            gr.update(value='Euler a'),  # ui_txt2img_sampler
            gr.update(value='Euler a'),  # ui_img2img_sampler
            gr.update(value='Automatic'),  # ui_txt2img_scheduler
            gr.update(value='Automatic'),  # ui_img2img_scheduler
        ]

    if shared.opts.forge_preset == 'xl':
        return [
            gr.update(visible=True),  # ui_vae
            gr.update(visible=False, value=1),  # ui_clip_skip
            gr.update(visible=True, value='Automatic'),  # ui_forge_unet_storage_dtype_options
            gr.update(visible=False, value='Queue'),  # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),  # ui_forge_pin_shared_memory
            gr.update(visible=False, value=total_vram - 1024),  # ui_forge_inference_memory
            gr.update(value=896),  # ui_txt2img_width
            gr.update(value=1024),  # ui_img2img_width
            gr.update(value=1152),  # ui_txt2img_height
            gr.update(value=1024),  # ui_img2img_height
            gr.update(value=5),  # ui_txt2img_cfg
            gr.update(value=5),  # ui_img2img_cfg
            gr.update(visible=False, value=3.5),  # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),  # ui_img2img_distilled_cfg
            gr.update(value='DPM++ 2M SDE'),  # ui_txt2img_sampler
            gr.update(value='DPM++ 2M SDE'),  # ui_img2img_sampler
            gr.update(value='Karras'),  # ui_txt2img_scheduler
            gr.update(value='Karras'),  # ui_img2img_scheduler
        ]

    if shared.opts.forge_preset == 'flux':
        return [
            gr.update(visible=True),  # ui_vae
            gr.update(visible=False, value=1),  # ui_clip_skip
            gr.update(visible=True, value='Automatic'),  # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),  # ui_forge_async_loading
            gr.update(visible=True, value='CPU'),  # ui_forge_pin_shared_memory
            gr.update(visible=True, value=total_vram - 1024),  # ui_forge_inference_memory
            gr.update(value=896),  # ui_txt2img_width
            gr.update(value=1024),  # ui_img2img_width
            gr.update(value=1152),  # ui_txt2img_height
            gr.update(value=1024),  # ui_img2img_height
            gr.update(value=1),  # ui_txt2img_cfg
            gr.update(value=1),  # ui_img2img_cfg
            gr.update(visible=True, value=3.5),  # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=3.5),  # ui_img2img_distilled_cfg
            gr.update(value='Euler'),  # ui_txt2img_sampler
            gr.update(value='Euler'),  # ui_img2img_sampler
            gr.update(value='Simple'),  # ui_txt2img_scheduler
            gr.update(value='Simple'),  # ui_img2img_scheduler
        ]

    return [
        gr.update(visible=True),  # ui_vae
        gr.update(visible=True, value=1),  # ui_clip_skip
        gr.update(visible=True, value='Automatic'),  # ui_forge_unet_storage_dtype_options
        gr.update(visible=True, value='Queue'),  # ui_forge_async_loading
        gr.update(visible=True, value='CPU'),  # ui_forge_pin_shared_memory
        gr.update(visible=True, value=total_vram - 1024),  # ui_forge_inference_memory
        gr.update(value=896),  # ui_txt2img_width
        gr.update(value=1024),  # ui_img2img_width
        gr.update(value=1152),  # ui_txt2img_height
        gr.update(value=1024),  # ui_img2img_height
        gr.update(value=7),  # ui_txt2img_cfg
        gr.update(value=7),  # ui_img2img_cfg
        gr.update(visible=True, value=3.5),  # ui_txt2img_distilled_cfg
        gr.update(visible=True, value=3.5),  # ui_img2img_distilled_cfg
        gr.update(value='DPM++ 2M'),  # ui_txt2img_sampler
        gr.update(value='DPM++ 2M'),  # ui_img2img_sampler
        gr.update(value='Automatic'),  # ui_txt2img_scheduler
        gr.update(value='Automatic'),  # ui_img2img_scheduler
    ]
