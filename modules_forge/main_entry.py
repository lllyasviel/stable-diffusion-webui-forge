import gradio as gr

from modules import shared_items, shared, ui_common, sd_models
from modules_forge import main_thread


sd_model_checkpoint: gr.Dropdown = None
sd_model_checkpoint_current_selection: str = shared.opts.sd_model_checkpoint
if sd_model_checkpoint_current_selection is None:
    sd_models.list_models()
    if len(sd_models.checkpoints_list) > 0:
        sd_model_checkpoint_current_selection = next(iter(sd_models.checkpoints_list.keys()))


sd_vae: gr.Dropdown = None


def make_checkpoint_manager_ui():
    global sd_model_checkpoint, sd_vae

    sd_model_checkpoint_args = lambda: {"choices": shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)}
    sd_model_checkpoint = gr.Dropdown(
        value=sd_model_checkpoint_current_selection,
        label="Checkpoint",
        **sd_model_checkpoint_args()
    )
    ui_common.create_refresh_button(sd_model_checkpoint, shared_items.refresh_checkpoints, sd_model_checkpoint_args, f"forge_refresh_checkpoint")

    sd_vae_args = lambda: {"choices": shared_items.sd_vae_items()}
    sd_vae = gr.Dropdown(
        value="Automatic",
        label="VAE",
        **sd_vae_args()
    )
    ui_common.create_refresh_button(sd_model_checkpoint, shared_items.refresh_vae_list, sd_vae_args, f"forge_refresh_vae")

    return


def checkpoint_change(ckpt_name):
    global sd_model_checkpoint_current_selection
    sd_model_checkpoint_current_selection = ckpt_name

    print(f'Checkpoint Selected: {ckpt_name}')
    shared.opts.set('sd_model_checkpoint', ckpt_name)
    shared.opts.save(shared.config_filename)

    sd_models.load_model(checkpoint_name=ckpt_name)
    return


def forge_main_entry():
    sd_model_checkpoint.change(lambda x: main_thread.async_run(checkpoint_change, x), inputs=[sd_model_checkpoint])
    return
