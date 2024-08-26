import torch
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

def patch_freeu_v2(unet_patcher, b1, b2, s1, s2):
    logger.info("Entering patch_freeu_v2 function")
    logger.info(f"unet_patcher type: {type(unet_patcher)}")
    
    # Debug: Print the structure of unet_patcher
    logger.info(f"unet_patcher attributes: {dir(unet_patcher)}")
    
    # Try to infer model_channels from the model structure
    model_channels = None
    
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

    # Try different attributes to find model_channels
    if hasattr(diffusion_model, 'in_channels'):
        model_channels = diffusion_model.in_channels
        logger.info(f"Found model_channels from in_channels: {model_channels}")
    elif hasattr(diffusion_model, 'config') and isinstance(diffusion_model.config, dict):
        model_channels = diffusion_model.config.get('model_channels', 
                                                    diffusion_model.config.get('channels', None))
        logger.info(f"Found model_channels from config: {model_channels}")
    
    if model_channels is None:
        # If we still can't find it, let's try to infer from the model structure
        for attr_name in dir(diffusion_model):
            attr = getattr(diffusion_model, attr_name)
            if isinstance(attr, torch.nn.Module):
                for param in attr.parameters():
                    if param.dim() > 1:
                        model_channels = param.shape[0]
                        logger.info(f"Inferred model_channels from {attr_name}: {model_channels}")
                        break
                if model_channels is not None:
                    break
    
    if model_channels is None:
        logger.warning("Could not infer model_channels. Using default value of 320.")
        model_channels = 320

    logger.info(f"Final model_channels value: {model_channels}")

    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
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