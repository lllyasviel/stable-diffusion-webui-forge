import gradio as gr

from modules import scripts
from ldm_patched.modules import model_management


class NeverOOMForForge(scripts.Script):
    sorting_priority = 18

    def __init__(self):
        self.previous_unet_enabled = False
        self.original_vram_state = model_management.vram_state

    def title(self):
        return "Never OOM Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            unet_enabled = gr.Checkbox(label='Enabled for UNet (always maximize offload)', value=False)
            vae_enabled = gr.Checkbox(label='Enabled for VAE (always tiled)', value=False)
        return unet_enabled, vae_enabled

    def process(self, p, *script_args, **kwargs):
        unet_enabled, vae_enabled = script_args

        if unet_enabled:
            print('NeverOOM Enabled for UNet (always maximize offload)')

        if vae_enabled:
            print('NeverOOM Enabled for VAE (always tiled)')

        model_management.VAE_ALWAYS_TILED = vae_enabled

        if self.previous_unet_enabled != unet_enabled:
            model_management.unload_all_models()
            if unet_enabled:
                self.original_vram_state = model_management.vram_state
                model_management.vram_state = model_management.VRAMState.NO_VRAM
            else:
                model_management.vram_state = self.original_vram_state
            print(f'VARM State Changed To {model_management.vram_state.name}')
            self.previous_unet_enabled = unet_enabled

        return
