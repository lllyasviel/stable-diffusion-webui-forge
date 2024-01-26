import gradio as gr
import os

from modules import scripts, script_callbacks
from modules.paths import models_path


svd_root = os.path.join(models_path, 'svd')
os.makedirs(svd_root, exist_ok=True)


class ForgeSVD(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "SVD"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as svd_block:
        with gr.Row():
            with gr.Column():
                width = gr.Slider(label="width", minimum=64, maximum=2048, value=512, step=64, interactive=True)
                height = gr.Slider(label="height", minimum=64, maximum=2048, value=512, step=64, interactive=True)

            with gr.Column():
                png_output = gr.Button(value="Save PNG")

    return [(svd_block, "SVD", "svd")]


script_callbacks.on_ui_tabs(on_ui_tabs)
