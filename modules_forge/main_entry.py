import torch
import gradio as gr

from modules import shared_items, shared, ui_common, sd_models, processing
from modules import sd_vae as sd_vae_module
from backend import memory_management, stream


total_vram = int(memory_management.total_vram)

ui_checkpoint: gr.Dropdown = None
ui_vae: gr.Dropdown = None
ui_clip_skip: gr.Slider = None

forge_unet_storage_dtype_options = {
    'None': None,
    'fp8e4m3': torch.float8_e4m3fn,
    'fp8e5m2': torch.float8_e5m2,
}


def bind_to_opts(comp, k, save=False, callback=None):
    def on_change(v):
        shared.opts.set(k, v)
        if save:
            shared.opts.save(shared.config_filename)
        if callback is not None:
            callback()
        return

    comp.change(on_change, inputs=[comp], show_progress=False)
    return


def make_checkpoint_manager_ui():
    global ui_checkpoint, ui_vae, ui_clip_skip

    if shared.opts.sd_model_checkpoint in [None, 'None', 'none', '']:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set('sd_model_checkpoint', next(iter(sd_models.checkpoints_list.keys())))

    sd_model_checkpoint_args = lambda: {"choices": shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)}
    ui_checkpoint = gr.Dropdown(
        value=shared.opts.sd_model_checkpoint,
        label="Checkpoint",
        elem_classes=['model_selection'],
        **sd_model_checkpoint_args()
    )
    ui_common.create_refresh_button(ui_checkpoint, shared_items.refresh_checkpoints, sd_model_checkpoint_args, f"forge_refresh_checkpoint")

    sd_vae_args = lambda: {"choices": shared_items.sd_vae_items()}
    ui_vae = gr.Dropdown(
        value=shared.opts.sd_vae,
        label="VAE",
        **sd_vae_args()
    )
    ui_common.create_refresh_button(ui_vae, shared_items.refresh_vae_list, sd_vae_args, f"forge_refresh_vae")

    ui_forge_unet_storage_dtype_options = gr.Radio(label="Diffusion in FP8", value=shared.opts.forge_unet_storage_dtype, choices=list(forge_unet_storage_dtype_options.keys()))
    bind_to_opts(ui_forge_unet_storage_dtype_options, 'forge_unet_storage_dtype', save=True, callback=refresh_model_loading_parameters)

    from backend.args import args as backend_args

    ui_forge_inference_memory = gr.Slider(label="Model Memory (MB)", value=total_vram - shared.opts.forge_inference_memory, minimum=0, maximum=int(memory_management.total_vram), step=1, visible=backend_args.i_am_lllyasviel)
    ui_forge_async_loading = gr.Checkbox(label="Async Loader", value=shared.opts.forge_async_loading, visible=backend_args.i_am_lllyasviel)
    ui_forge_pin_shared_memory = gr.Checkbox(label="Offload to Shared Memory", value=shared.opts.forge_pin_shared_memory, visible=backend_args.i_am_lllyasviel)

    mem_comps = [ui_forge_inference_memory, ui_forge_async_loading, ui_forge_pin_shared_memory]

    ui_forge_inference_memory.change(refresh_memory_management_settings, inputs=mem_comps)
    ui_forge_async_loading.change(refresh_memory_management_settings, inputs=mem_comps)
    ui_forge_pin_shared_memory.change(refresh_memory_management_settings, inputs=mem_comps)

    ui_clip_skip = gr.Slider(label="Clip skip", value=shared.opts.CLIP_stop_at_last_layers, **{"minimum": 1, "maximum": 12, "step": 1})
    bind_to_opts(ui_clip_skip, 'CLIP_stop_at_last_layers', save=False)

    return


def refresh_memory_management_settings(model_memory, async_loading, pin_shared_memory):
    inference_memory = total_vram - model_memory

    shared.opts.set('forge_async_loading', async_loading)
    shared.opts.set('forge_inference_memory', inference_memory)
    shared.opts.set('forge_pin_shared_memory', pin_shared_memory)

    stream.stream_activated = async_loading
    memory_management.current_inference_memory = inference_memory * 1024 * 1024
    memory_management.PIN_SHARED_MEMORY = pin_shared_memory

    print(f'Stream Set to: {stream.stream_activated}')
    print(f'Stream Used by CUDA: {stream.should_use_stream()}')
    print(f'Current Inference Memory: {memory_management.minimum_inference_memory() / (1024 * 1024):.2f} MB')
    print(f'PIN Shared Memory: {pin_shared_memory}')

    processing.need_global_unload = True
    return


def refresh_model_loading_parameters():
    from modules.sd_models import select_checkpoint, model_data

    checkpoint_info = select_checkpoint()
    vae_resolution = sd_vae_module.resolve_vae(checkpoint_info.filename)

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        vae_filename=vae_resolution.vae,
        unet_storage_dtype=forge_unet_storage_dtype_options[shared.opts.forge_unet_storage_dtype]
    )

    print(f'Loading parameters: {model_data.forge_loading_parameters}')

    return


def checkpoint_change(ckpt_name):
    shared.opts.set('sd_model_checkpoint', ckpt_name)
    shared.opts.save(shared.config_filename)

    refresh_model_loading_parameters()
    return


def vae_change(vae_name):
    shared.opts.set('sd_vae', vae_name)

    refresh_model_loading_parameters()
    return


def forge_main_entry():
    ui_checkpoint.change(checkpoint_change, inputs=[ui_checkpoint], show_progress=False)
    ui_vae.change(vae_change, inputs=[ui_vae], show_progress=False)

    # Load Model
    refresh_model_loading_parameters()
    return
