import torch
import torch.nn.functional as F
import gradio as gr
from modules import scripts
import logging
import sys
from pathlib import Path

def setup_logging(log_file=None):
    logger = logging.getLogger("FreeU")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

logger = setup_logging(log_file=Path("freeu_log.txt"))

def Fourier_filter(x, threshold, scale):
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)
    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered.to(x.dtype)

def is_flux_model(model):
    return 'IntegratedFluxTransformer2DModel' in str(type(model))

def apply_freeu_to_flux(h, b1, b2, s1, s2):
    logger.info(f"Applying FreeU to FLUX. Input shape: {h.shape}")
    
    # Global scaling
    h = h * b1
    logger.info(f"After global scaling. Mean: {h.mean().item()}, Std: {h.std().item()}")
    
    # Apply Fourier filter
    h_freq = torch.fft.fftn(h, dim=(-2, -1))
    h_freq = torch.fft.fftshift(h_freq, dim=(-2, -1))
    
    B, C, H, W = h_freq.shape
    mask = torch.ones_like(h_freq)
    crow, ccol = H // 2, W // 2
    mask[..., crow-H//4:crow+H//4, ccol-W//4:ccol+W//4] = s1
    
    h_freq = h_freq * mask
    h_freq = torch.fft.ifftshift(h_freq, dim=(-2, -1))
    h = torch.fft.ifftn(h_freq, dim=(-2, -1)).real
    logger.info(f"After Fourier filter. Mean: {h.mean().item()}, Std: {h.std().item()}")
    
    # Non-linear transformation
    h = torch.tanh(h * b2) * s2
    logger.info(f"After non-linear transform. Mean: {h.mean().item()}, Std: {h.std().item()}")
    
    logger.info(f"FreeU applied to FLUX. Output shape: {h.shape}")
    return h

def patch_freeu_v2(unet_patcher, b1, b2, s1, s2):
    logger.info("Entering patch_freeu_v2 function")
    logger.info(f"unet_patcher type: {type(unet_patcher)}")
    
    if hasattr(unet_patcher, 'model'):
        diffusion_model = unet_patcher.model
        logger.info("unet_patcher has 'model' attribute")
    else:
        diffusion_model = unet_patcher
        logger.info("Using unet_patcher as diffusion_model")
    
    logger.info(f"diffusion_model type: {type(diffusion_model)}")
    logger.info(f"diffusion_model attributes: {dir(diffusion_model)}")
    
    if is_flux_model(diffusion_model):
        logger.info("FLUX model detected. Applying FLUX-specific FreeU.")
        
        def flux_output_block_patch(h, *args, **kwargs):
            logger.info(f"Applying FreeU to FLUX output. Input shape: {h.shape}")
            result = apply_freeu_to_flux(h, b1, b2, s1, s2)
            logger.info(f"FreeU applied to FLUX output. Output shape: {result.shape}")
            return result, None

        if hasattr(unet_patcher, 'set_model_output_block_patch'):
            unet_patcher.set_model_output_block_patch(flux_output_block_patch)
            logger.info("FLUX-specific FreeU patch applied successfully.")
        else:
            logger.warning("Could not set output block patch for FLUX model. FreeU may not be applied.")
        
        return unet_patcher
    
    # Handling for standard U-Net models
    if not hasattr(diffusion_model, 'input_blocks') or not hasattr(diffusion_model, 'output_blocks'):
        logger.warning("Model architecture is not compatible with standard FreeU. Attempting generic approach.")
        
        def generic_output_block_patch(h, *args, **kwargs):
            return apply_freeu_to_flux(h, b1, b2, s1, s2), None

        if hasattr(unet_patcher, 'set_model_output_block_patch'):
            unet_patcher.set_model_output_block_patch(generic_output_block_patch)
            logger.info("Generic FreeU patch applied successfully.")
        else:
            logger.warning("Could not set output block patch. FreeU may not be applied.")
        
        return unet_patcher

    # Standard U-Net FreeU implementation
    model_channels = getattr(diffusion_model, 'model_channels', 320)
    logger.info(f"model_channels: {model_channels}")

    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    logger.info(f"scale_dict: {scale_dict}")

    def output_block_patch(h, hsp, transformer_options):
        scale = scale_dict.get(h.shape[1], None)
        if scale is not None:
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
            h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)
            hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
        return h, hsp

    m = unet_patcher.clone()
    m.set_model_output_block_patch(output_block_patch)
    logger.info("Standard U-Net FreeU patch applied successfully.")
    return m

class FreeUForForge(scripts.Script):
    sorting_priority = 12

    def title(self):
        return "FreeU Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title(),
                          elem_id="extensions-freeu",
                          elem_classes=["extensions-freeu"]):
            freeu_enabled = gr.Checkbox(label='Enabled', value=False)
            freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
            freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
            freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
            freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)

        return freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2 = script_args

        if not freeu_enabled:
            return

        unet = p.sd_model.forge_objects.unet
        
        logger.info(f"Model type before FreeU: {type(unet)}")

        try:
            unet = patch_freeu_v2(unet, freeu_b1, freeu_b2, freeu_s1, freeu_s2)
            logger.info(f"FreeU applied. Model type after: {type(unet)}")
        except Exception as e:
            logger.error(f"Error in patch_freeu_v2: {str(e)}", exc_info=True)
            return

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            freeu_enabled=freeu_enabled,
            freeu_b1=freeu_b1,
            freeu_b2=freeu_b2,
            freeu_s1=freeu_s1,
            freeu_s2=freeu_s2,
        ))

        logger.info("FreeU parameters applied successfully")
        return
