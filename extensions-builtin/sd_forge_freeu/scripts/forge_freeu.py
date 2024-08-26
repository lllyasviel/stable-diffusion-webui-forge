import torch
import torch.nn.functional as F
import gradio as gr
from modules import scripts
import logging
import sys
from pathlib import Path

def setup_logging(log_file=None):
    # Create a logger
    logger = logging.getLogger("FreeU")
    logger.setLevel(logging.DEBUG)

    # Create console handler and set level to debug
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # If a log file is specified, add a file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Setup logging (you can specify a log file path if needed)
logger = setup_logging(log_file=Path("freeu_log.txt"))

def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)

def infer_model_channels(diffusion_model):
    if is_flux_model(diffusion_model):
        logger.info("Flux model detected. Using default channels.")
        return 320  # Or another appropriate default for Flux
    
    if hasattr(diffusion_model, 'in_channels'):
        return diffusion_model.in_channels
    elif hasattr(diffusion_model, 'model') and hasattr(diffusion_model.model, 'diffusion_model'):
        return diffusion_model.model.diffusion_model.in_channels
    elif hasattr(diffusion_model, 'config'):
        if isinstance(diffusion_model.config, dict):
            return diffusion_model.config.get('model_channels', diffusion_model.config.get('channels'))
    
    # If we still can't find it, let's try to infer from the model structure
    for attr_name in dir(diffusion_model):
        attr = getattr(diffusion_model, attr_name)
        if isinstance(attr, torch.nn.Module):
            for param in attr.parameters():
                if param.dim() > 1:
                    return param.shape[0]
    
    logger.warning("Could not infer model_channels. Using default value of 320.")
    return 320

def is_flux_model(model):
    return 'flux' in str(type(model)).lower() or hasattr(model, 'flux_attributes')  # Add any other FLUX-specific attributes

def is_compatible_architecture(model):
    if is_flux_model(model):
        logger.info("Flux model detected. Assuming compatibility.")
        return True
    # Check for U-Net like structure
    return hasattr(model, 'input_blocks') and hasattr(model, 'output_blocks')

def apply_freeu_to_flux(h, scale_dict, b1, b2, s1, s2):
    logger.info(f"Applying FreeU to FLUX. Input shape: {h.shape}")
    
    B, C, H, W = h.shape
    
    # More aggressive scaling for FLUX
    h = h * b1  # Global scaling
    
    # Apply Fourier filter
    h_freq = torch.fft.fftn(h, dim=(-2, -1))
    h_freq = torch.fft.fftshift(h_freq, dim=(-2, -1))
    
    mask = torch.ones_like(h_freq)
    crow, ccol = H // 2, W // 2
    mask[..., crow-1:crow+1, ccol-1:ccol+1] = s1
    
    h_freq = h_freq * mask
    h_freq = torch.fft.ifftshift(h_freq, dim=(-2, -1))
    h = torch.fft.ifftn(h_freq, dim=(-2, -1)).real
    
    # Additional contrast adjustment
    h_mean = h.mean()
    h = (h - h_mean) * b2 + h_mean
    
    logger.info(f"FreeU applied to FLUX. Output shape: {h.shape}")
    return h

def flow_matching_scale(h, b1, b2):
    # This function attempts to scale the features based on the flow matching principle
    # Note: This is a simplified approximation and may need adjustment based on FLUX's exact implementation
    B, C, H, W = h.shape
    
    # Create a flow field
    flow = torch.randn(B, 2, H, W, device=h.device)
    flow = F.normalize(flow, dim=1)
    
    # Scale the flow field
    flow = flow * b1
    
    # Apply the flow to the features
    grid = F.affine_grid(flow.permute(0, 2, 3, 1).view(B, H * W, 2), h.size())
    h_warped = F.grid_sample(h, grid, mode='bilinear', padding_mode='border')
    
    # Combine original and warped features
    h_combined = h + b2 * (h_warped - h)
    
    return h_combined

def patch_freeu_v2(unet_patcher, b1, b2, s1, s2):
    logger.info("Entering patch_freeu_v2 function")
    logger.info(f"unet_patcher type: {type(unet_patcher)}")
    
    # Debug: Print the structure of unet_patcher
    logger.info(f"unet_patcher attributes: {dir(unet_patcher)}")
    
    if is_flux_model(unet_patcher):
        logger.info("FLUX model detected. Using specialized handling.")
        
        def flux_output_block_patch(h, *args, **kwargs):
            return apply_freeu_to_flux(h, None, b1, b2, s1, s2), None
        
        if hasattr(unet_patcher, 'set_model_output_block_patch'):
            unet_patcher.set_model_output_block_patch(flux_output_block_patch)
        else:
            logger.warning("Could not set output block patch for FLUX model. FreeU may not be applied.")
        
        return unet_patcher
    
    if hasattr(unet_patcher, 'model'):
        logger.info("unet_patcher has 'model' attribute")
        if hasattr(unet_patcher.model, 'diffusion_model'):
            logger.info("unet_patcher.model has 'diffusion_model' attribute")
            diffusion_model = unet_patcher.model.diffusion_model
        else:
            logger.info("Using unet_patcher.model as diffusion_model")
            diffusion_model = unet_patcher.model
    else:
        logger.info("Using unet_patcher as diffusion_model")
        diffusion_model = unet_patcher

    logger.info(f"diffusion_model type: {type(diffusion_model)}")
    logger.info(f"diffusion_model attributes: {dir(diffusion_model)}")

    if not is_compatible_architecture(diffusion_model):
        logger.warning("Model architecture is not compatible with FreeU. Skipping application.")
        return unet_patcher

    if 'gguf' in str(type(diffusion_model)):
        logger.warning("Quantized model detected. FreeU may not be directly applicable.")
        return unet_patcher

    model_channels = infer_model_channels(diffusion_model)

    if model_channels is None or model_channels > 1000000:  # Arbitrary large number to catch unreasonable values
        logger.warning("Could not reliably infer model channels. FreeU will be disabled for this run.")
        return unet_patcher

    logger.info(f"Final model_channels value: {model_channels}")

    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        if is_flux_model(h):
            return apply_freeu_to_flux(h, scale_dict, b1, b2, s1, s2), hsp

        scale = scale_dict.get(h.shape[1], None)
        if scale is not None:
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

            h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)

            if hsp.device not in on_cpu_devices:
                try:
                    hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                except:
                    logger.warning(f"Device {hsp.device} does not support the torch.fft functions used in the FreeU node, switching to CPU.")
                    on_cpu_devices[hsp.device] = True
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
            else:
                hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

        return h, hsp

    m = unet_patcher.clone()
    m.set_model_output_block_patch(output_block_patch)
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
