import gradio as gr

from modules import shared_items, shared, ui_common


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


def forge_main_entry():
    sd_model_checkpoint.change(lambda x: print(x), inputs=[sd_model_checkpoint])
    return
