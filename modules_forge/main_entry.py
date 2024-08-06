import gradio as gr

from modules import shared_items, shared, ui_common, sd_models
from modules_forge import main_thread


sd_model_checkpoint: gr.Dropdown = None


def make_checkpoint_manager_ui():
    global sd_model_checkpoint
    sd_model_checkpoint_args = lambda: {"choices": shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)}
    sd_model_checkpoint = gr.Dropdown(
        value=None,
        label="Checkpoint",
        **sd_model_checkpoint_args()
    )
    ui_common.create_refresh_button(sd_model_checkpoint, shared_items.refresh_checkpoints, sd_model_checkpoint_args, f"forge_refresh_checkpoint")

    return


def checkpoint_change(ckpt_name):
    a = 0
    # sd_models.reload_model_weights()
    return


def forge_main_entry():
    sd_model_checkpoint.change(main_thread.run_and_wait_result(checkpoint_change), inputs=[sd_model_checkpoint])
    return
