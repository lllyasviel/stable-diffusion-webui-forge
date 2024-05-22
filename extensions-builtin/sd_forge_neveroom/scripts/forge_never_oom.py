import gradio as gr
import torch
import modules.devices as devices

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
    
    """
    The following two functions are pulle directly from
    pkuliyi2015/multidiffusion-upscaler-for-automatic1111
    """
    def get_rcmd_enc_tsize(self):
        if torch.cuda.is_available() and devices.device not in ['cpu', devices.cpu]:
            total_memory = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
            if   total_memory > 16*1000: ENCODER_TILE_SIZE = 3072
            elif total_memory > 12*1000: ENCODER_TILE_SIZE = 2048
            elif total_memory >  8*1000: ENCODER_TILE_SIZE = 1536
            else:                        ENCODER_TILE_SIZE = 960
        else:                            ENCODER_TILE_SIZE = 512
        return ENCODER_TILE_SIZE

    def get_rcmd_dec_tsize(self):
        if torch.cuda.is_available() and devices.device not in ['cpu', devices.cpu]:
            total_memory = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
            if   total_memory > 30*1000: DECODER_TILE_SIZE = 256
            elif total_memory > 16*1000: DECODER_TILE_SIZE = 192
            elif total_memory > 12*1000: DECODER_TILE_SIZE = 128
            elif total_memory >  8*1000: DECODER_TILE_SIZE = 96
            else:                        DECODER_TILE_SIZE = 64
        else:                            DECODER_TILE_SIZE = 64
        return DECODER_TILE_SIZE

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            unet_enabled = gr.Checkbox(label='Enabled for UNet (always maximize offload)', value=False)
            vae_enabled = gr.Checkbox(label='Enabled for VAE (always tiled)', value=False)
            encoder_tile_size = gr.Slider(label='Encoder Tile Size', minimum=256, maximum=4096, step=16, value=self.get_rcmd_enc_tsize())
            decoder_tile_size = gr.Slider(label='Decoder Tile Size', minimum=48,  maximum=512,  step=16, value=self.get_rcmd_dec_tsize())
        return unet_enabled, vae_enabled, encoder_tile_size, decoder_tile_size

    def process(self, p, *script_args, **kwargs):
        unet_enabled, vae_enabled, encoder_tile_size, decoder_tile_size = script_args

        if unet_enabled:
            print('NeverOOM Enabled for UNet (always maximize offload)')

        if vae_enabled:
            print('NeverOOM Enabled for VAE (always tiled)')
            print('With tile sizes')
            print(f'Encode:\t x:{encoder_tile_size}\t y:{encoder_tile_size}')
            print(f'Decode:\t x:{decoder_tile_size}\t y:{decoder_tile_size}')

        model_management.VAE_ALWAYS_TILED = vae_enabled
        model_management.VAE_ENCODE_TILE_SIZE_X = model_management.VAE_ENCODE_TILE_SIZE_Y = encoder_tile_size
        model_management.VAE_DECODE_TILE_SIZE_X = model_management.VAE_DECODE_TILE_SIZE_Y = decoder_tile_size

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
