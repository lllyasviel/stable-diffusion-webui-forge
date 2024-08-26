import torch
import torch.nn.functional as F
import gradio as gr
from modules import scripts
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    if 'flux' in str(type(diffusion_model)).lower():
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

def is_compatible_architecture(model):
    if 'flux' in str(type(model)).lower():
        logger.info("Flux model detected. Assuming compatibility.")
        return True
    # Check for U-Net like structure
    return hasattr(model, 'input_blocks') and hasattr(model, 'output_blocks')

def apply_freeu_to_flux(h, scale_dict, b1, b2, s1, s2):
    # Assuming 'h' is the output of a transformer block or a combination of transformer and diffusion outputs
    B, C, H, W = h.shape
    
    # Split the channels into two halves
    h1, h2 = torch.split(h, C // 2, dim=1)
    
    # Apply scaling to the first half (assumed to be more related to global structure)
    if C in scale_dict:
        scale = scale_dict[C]
        h1_mean = h1.mean(1, keepdim=True)
        h1_max, _ = torch.max(h1_mean.view(B, -1), dim=-1, keepdim=True)
        h1_min, _ = torch.min(h1_mean.view(B, -1), dim=-1, keepdim=True)
        h1_mean = (h1_mean - h1_min.view(B, 1, 1, 1)) / (h1_max - h1_min).view(B, 1, 1, 1)
        h1 = h1 * ((scale[0] - 1) * h1_mean + 1)
    
    # Apply Fourier filter to the second half (assumed to be more related to fine details)
    h2 = Fourier_filter(h2, threshold=1, scale=s1)
    
    # Combine the two halves
    h_combined = torch.cat([h1, h2], dim=1)
    
    # Apply additional scaling based on the flow matching principle
    h_flow = flow_matching_scale(h_combined, b1, b2)
    
    return h_flow

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

    if 'flux' in str(type(diffusion_model)).lower():
        logger.info("Flux model detected. Using specialized handling.")
        try:
            model_channels = infer_model_channels(diffusion_model)
        except Exception as e:
            logger.warning(f"Unable to apply FreeU to FLUX model: {str(e)}. Returning original model.")
            return unet_patcher
    else:
        model_channels = infer_model_channels(diffusion_model)

    if model_channels is None or model_channels > 1000000:  # Arbitrary large number to catch unreasonable values
        logger.warning("Could not reliably infer model channels. FreeU will be disabled for this run.")
        return unet_patcher

    logger.info(f"Final model_channels value: {model_channels}")

    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        if 'flux' in str(type(h)).lower():
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

        try:
            unet = patch_freeu_v2(unet, freeu_b1, freeu_b2, freeu_s1, freeu_s2)
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

        return
