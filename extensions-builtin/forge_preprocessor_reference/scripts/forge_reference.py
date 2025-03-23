import torch
from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from backend.sampling.sampling_function import sampling_function_inner
from backend import attention

# Utility Functions
def sdp(q, k, v, transformer_options):
    """Scaled dot-product attention with head support."""
    if q.shape[0] == 0:
        return q
    return attention.attention_function(q, k, v, heads=transformer_options["n_heads"], mask=None)

def adain(x, target_std, target_mean):
    """Adaptive Instance Normalization."""
    if x.shape[0] == 0:
        return x
    std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True, correction=0)
    return (((x - mean) / (std + 1e-8)) * target_std) + target_mean  # Added epsilon for stability

def zero_cat(a, b, dim):
    """Concatenate tensors, handling empty cases."""
    if a.shape[0] == 0:
        return b
    if b.shape[0] == 0:
        return a
    return torch.cat([a, b], dim=dim)

class PreprocessorReference(Preprocessor):
    """
    A preprocessor for reference-based processing in a generative pipeline, supporting attention and AdaIN mechanisms.
    Enhancements include modularity, multiple attention layers, dynamic style fidelity, and robust error handling.
    """
    def __init__(
        self,
        name: str,
        attn_layers: list[str] = ['attn1'],  # Support for multiple attention layers
        use_adain: bool = True,
        priority: int = 0,
        cache_style: bool = False  # New feature: cache style for performance
    ):
        super().__init__()
        # Core Attributes
        self.name = name
        self.attn_layers = attn_layers  # e.g., ['attn1', 'attn2']
        self.use_adain = use_adain
        self.sorting_priority = priority
        self.cache_style = cache_style

        # UI Parameters
        self.tags = ['Reference', 'Style Transfer']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.slider_1 = PreprocessorParameter(
            label='Style Fidelity', value=0.5, minimum=0.0, maximum=1.0, step=0.01, visible=True
        )
        self.slider_2 = PreprocessorParameter(  # New: Attention strength
            label='Attention Strength', value=1.0, minimum=0.0, maximum=2.0, step=0.01, visible=True
        )
        self.slider_3 = PreprocessorParameter(  # New: AdaIN strength
            label='AdaIN Strength', value=1.0, minimum=0.0, maximum=2.0, step=0.01, visible=True
        )
        self.show_control_mode = False
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = False
        self.do_not_need_model = True

        # State Variables
        self.is_recording_style = False
        self.recorded_attn = {layer: {} for layer in attn_layers}  # Multi-layer attention storage
        self.recorded_h = {}
        self.latent_image = None
        self.gen_cpu = None
        self.sigma_min = 0.0
        self.sigma_max = 0.0
        self.weight = 0.0
        self.style_fidelity = 0.0
        self.attn_strength = 1.0  # New parameter
        self.adain_strength = 1.0  # New parameter
        self.style_cache_key = None  # For caching

    def get_sigma_range(self, unet, start_percent: float, end_percent: float) -> tuple[float, float]:
        """Calculate sigma range for guidance based on percentages."""
        sigma_max = unet.model.predictor.percent_to_sigma(start_percent)
        sigma_min = unet.model.predictor.percent_to_sigma(end_percent)
        return sigma_min, sigma_max

    def conditioning_modifier(self, model, x, timestep, uncond, cond, cond_scale, model_options, seed):
        """Record style by running a sampling step with noisy latent image."""
        sigma = timestep[0].item()
        if not (self.sigma_min <= sigma <= self.sigma_max):
            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        if self.cache_style and self.style_cache_key == (cond.tobytes(), seed):
            # Skip recording if style is cached
            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        self.is_recording_style = True
        xt = self.latent_image.to(x.device, x.dtype) + torch.randn(
            x.size(), dtype=x.dtype, generator=self.gen_cpu, device=x.device
        ) * sigma
        sampling_function_inner(model, xt, timestep, uncond, cond, 1, model_options, seed)
        self.is_recording_style = False

        if self.cache_style:
            self.style_cache_key = (cond.tobytes(), seed)  # Cache based on condition and seed

        return model, x, timestep, uncond, cond, cond_scale, model_options, seed

    def attn_proc(self, layer: str, q, k, v, transformer_options):
        """Generalized attention processor for any specified layer."""
        if layer not in self.attn_layers:
            return sdp(q, k, v, transformer_options)

        sigma = transformer_options["sigmas"][0].item()
        if not (self.sigma_min <= sigma <= self.sigma_max):
            return sdp(q, k, v, transformer_options)

        location = (
            transformer_options['block'][0],
            transformer_options['block'][1],
            transformer_options['block_index']
        )
        channel = int(q.shape[2])
        minimal_channel = 1500 - 1280 * self.weight

        if channel < minimal_channel:
            return sdp(q, k, v, transformer_options)

        if self.is_recording_style:
            self.recorded_attn[layer][location] = (k.clone(), v.clone())  # Clone to avoid inplace issues
            return sdp(q, k, v, transformer_options)
        else:
            cond_indices = transformer_options['cond_indices']
            uncond_indices = transformer_options['uncond_indices']
            cond_or_uncond = transformer_options['cond_or_uncond']

            q_c, q_uc = q[cond_indices], q[uncond_indices]
            k_c, k_uc = k[cond_indices], k[uncond_indices]
            v_c, v_uc = v[cond_indices], v[uncond_indices]

            k_r, v_r = self.recorded_attn[layer].get(location, (k_c.new_zeros(k_c.shape), v_c.new_zeros(v_c.shape)))

            # Apply attention strength
            k_r = k_r * self.attn_strength
            v_r = v_r * self.attn_strength

            o_c = sdp(q_c, zero_cat(k_c, k_r, dim=1), zero_cat(v_c, v_r, dim=1), transformer_options)
            o_uc_strong = sdp(q_uc, k_uc, v_uc, transformer_options)
            o_uc_weak = sdp(q_uc, zero_cat(k_uc, k_r, dim=1), zero_cat(v_uc, v_r, dim=1), transformer_options)
            o_uc = o_uc_weak + (o_uc_strong - o_uc_weak) * self.style_fidelity

            recon = [o_c if cx == 0 else o_uc for cx in cond_or_uncond]
            return torch.cat(recon, dim=0)

    def block_proc(self, h, flag, transformer_options):
        """Apply AdaIN to hidden states with configurable strength."""
        if not self.use_adain or flag != 'after':
            return h

        sigma = transformer_options["sigmas"][0].item()
        if not (self.sigma_min <= sigma <= self.sigma_max):
            return h

        location = transformer_options['block']
        channel = int(h.shape[1])
        minimal_channel = 1500 - 1000 * self.weight

        if channel < minimal_channel:
            return h

        if self.is_recording_style:
            self.recorded_h[location] = torch.std_mean(h, dim=(2, 3), keepdim=True, correction=0)
            return h
        else:
            cond_indices = transformer_options['cond_indices']
            uncond_indices = transformer_options['uncond_indices']
            cond_or_uncond = transformer_options['cond_or_uncond']
            r_std, r_mean = self.recorded_h.get(location, (h.new_ones(h.shape[1:]), h.new_zeros(h.shape[1:])))

            h_c, h_uc = h[cond_indices], h[uncond_indices]
            o_c = adain(h_c, r_std * self.adain_strength, r_mean * self.adain_strength)
            o_uc_strong = h_uc
            o_uc_weak = adain(h_uc, r_std * self.adain_strength, r_mean * self.adain_strength)
            o_uc = o_uc_weak + (o_uc_strong - o_uc_weak) * self.style_fidelity

            recon = [o_c if cx == 0 else o_uc for cx in cond_or_uncond]
            return torch.cat(recon, dim=0)

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        """
        Prepare the preprocessor for each sampling step, setting up model modifiers and style recording.
        """
        unit = kwargs['unit']
        self.weight = float(unit.weight)
        self.style_fidelity = float(unit.threshold_a)
        self.attn_strength = float(unit.slider_2 if hasattr(unit, 'slider_2') else 1.0)
        self.adain_strength = float(unit.slider_3 if hasattr(unit, 'slider_3') else 1.0)
        start_percent = float(unit.guidance_start)
        end_percent = float(unit.guidance_end)

        # Model-specific adjustments
        if process.sd_model.is_sdxl:
            self.style_fidelity **= 3.0  # Adjust for SDXL sensitivity

        # Prepare latent image and generator
        vae = process.sd_model.forge_objects.vae
        self.latent_image = vae.encode(cond.movedim(1, -1))
        self.latent_image = process.sd_model.forge_objects.vae.first_stage_model.process_in(self.latent_image)

        gen_seed = process.seeds[0] + 1
        self.gen_cpu = torch.Generator().manual_seed(gen_seed)

        # Clone UNet and set sigma range
        unet = process.sd_model.forge_objects.unet.clone()
        self.sigma_min, self.sigma_max = self.get_sigma_range(unet, start_percent, end_percent)

        # Reset style records unless caching
        if not (self.cache_style and self.style_cache_key):
            for layer in self.attn_layers:
                self.recorded_attn[layer] = {}
            self.recorded_h = {}

        # Set up model modifiers
        unet.add_conditioning_modifier(self.conditioning_modifier)
        unet.add_block_modifier(self.block_proc)
        for layer in self.attn_layers:
            unet.set_model_replace_all(lambda *args, **kwargs: self.attn_proc(layer, *args, **kwargs), layer)

        process.sd_model.forge_objects.unet = unet
        return cond, mask

# Register Preprocessor Variants
add_supported_preprocessor(PreprocessorReference(
    name='reference_only',
    attn_layers=['attn1'],
    use_adain=False,
    priority=100,
    cache_style=False
))

add_supported_preprocessor(PreprocessorReference(
    name='reference_adain',
    attn_layers=[],
    use_adain=True,
    cache_style=False
))

add_supported_preprocessor(PreprocessorReference(
    name='reference_adain+attn',
    attn_layers=['attn1'],
    use_adain=True,
    cache_style=True  # Enable caching for combined mode
))
