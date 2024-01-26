import gradio as gr
import os

from modules import scripts, script_callbacks
from modules.paths import models_path
from modules import shared


svd_root = os.path.join(models_path, 'svd')
os.makedirs(svd_root, exist_ok=True)
svd_filenames = []


def update_svd_filenames():
    global svd_filenames
    svd_filenames = list(shared.walk_files(svd_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    return svd_filenames


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


update_svd_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
