import gradio as gr
from modules import scripts

from lib_latent_modifier.sampler_mega_modifier import ModelSamplerLatentMegaModifier

opModelSamplerLatentMegaModifier = ModelSamplerLatentMegaModifier().mega_modify


class LatentModifierForForge(scripts.Script):
    sorting_priority = 15

    def title(self):
        return "LatentModifier Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            sharpness_multiplier = gr.Slider(label='Sharpness Multiplier', minimum=-100.0, maximum=100.0, step=0.1,
                                             value=0.0)
            sharpness_method = gr.Radio(label='Sharpness Method',
                                        choices=['anisotropic', 'joint-anisotropic', 'gaussian', 'cas'],
                                        value='anisotropic')
            tonemap_multiplier = gr.Slider(label='Tonemap Multiplier', minimum=0.0, maximum=100.0, step=0.01, value=0.0)
            tonemap_method = gr.Radio(label='Tonemap Method',
                                      choices=['reinhard', 'reinhard_perchannel', 'arctan', 'quantile', 'gated',
                                               'cfg-mimic', 'spatial-norm'], value='reinhard')
            tonemap_percentile = gr.Slider(label='Tonemap Percentile', minimum=0.0, maximum=100.0, step=0.005,
                                           value=100.0)
            contrast_multiplier = gr.Slider(label='Contrast Multiplier', minimum=-100.0, maximum=100.0, step=0.1,
                                            value=0.0)
            combat_method = gr.Radio(label='Combat Method',
                                     choices=['subtract', 'subtract_channels', 'subtract_median', 'sharpen'],
                                     value='subtract')
            combat_cfg_drift = gr.Slider(label='Combat Cfg Drift', minimum=-10.0, maximum=10.0, step=0.01, value=0.0)
            rescale_cfg_phi = gr.Slider(label='Rescale Cfg Phi', minimum=-10.0, maximum=10.0, step=0.01, value=0.0)
            extra_noise_type = gr.Radio(label='Extra Noise Type',
                                        choices=['gaussian', 'uniform', 'perlin', 'pink', 'green', 'pyramid'],
                                        value='gaussian')
            extra_noise_method = gr.Radio(label='Extra Noise Method',
                                          choices=['add', 'add_scaled', 'speckle', 'cads', 'cads_rescaled',
                                                   'cads_speckle', 'cads_speckle_rescaled'], value='add')
            extra_noise_multiplier = gr.Slider(label='Extra Noise Multiplier', minimum=0.0, maximum=100.0, step=0.1,
                                               value=0.0)
            extra_noise_lowpass = gr.Slider(label='Extra Noise Lowpass', minimum=0, maximum=1000, step=1, value=100)
            divisive_norm_size = gr.Slider(label='Divisive Norm Size', minimum=1, maximum=255, step=1, value=127)
            divisive_norm_multiplier = gr.Slider(label='Divisive Norm Multiplier', minimum=0.0, maximum=1.0, step=0.01,
                                                 value=0.0)
            spectral_mod_mode = gr.Radio(label='Spectral Mod Mode', choices=['hard_clamp', 'soft_clamp'],
                                         value='hard_clamp')
            spectral_mod_percentile = gr.Slider(label='Spectral Mod Percentile', minimum=0.0, maximum=50.0, step=0.01,
                                                value=5.0)
            spectral_mod_multiplier = gr.Slider(label='Spectral Mod Multiplier', minimum=-15.0, maximum=15.0, step=0.01,
                                                value=0.0)
            affect_uncond = gr.Radio(label='Affect Uncond', choices=['None', 'Sharpness'], value='None')
            dyn_cfg_augmentation = gr.Radio(label='Dyn Cfg Augmentation',
                                            choices=['None', 'dyncfg-halfcosine', 'dyncfg-halfcosine-mimic'],
                                            value='None')

        return enabled, sharpness_multiplier, sharpness_method, tonemap_multiplier, tonemap_method, tonemap_percentile, contrast_multiplier, combat_method, combat_cfg_drift, rescale_cfg_phi, extra_noise_type, extra_noise_method, extra_noise_multiplier, extra_noise_lowpass, divisive_norm_size, divisive_norm_multiplier, spectral_mod_mode, spectral_mod_percentile, spectral_mod_multiplier, affect_uncond, dyn_cfg_augmentation

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        enabled, sharpness_multiplier, sharpness_method, tonemap_multiplier, tonemap_method, tonemap_percentile, contrast_multiplier, combat_method, combat_cfg_drift, rescale_cfg_phi, extra_noise_type, extra_noise_method, extra_noise_multiplier, extra_noise_lowpass, divisive_norm_size, divisive_norm_multiplier, spectral_mod_mode, spectral_mod_percentile, spectral_mod_multiplier, affect_uncond, dyn_cfg_augmentation = script_args

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = opModelSamplerLatentMegaModifier(unet, sharpness_multiplier, sharpness_method, tonemap_multiplier, tonemap_method, tonemap_percentile, contrast_multiplier, combat_method, combat_cfg_drift, rescale_cfg_phi, extra_noise_type, extra_noise_method, extra_noise_multiplier, extra_noise_lowpass, divisive_norm_size, divisive_norm_multiplier, spectral_mod_mode, spectral_mod_percentile, spectral_mod_multiplier, affect_uncond, dyn_cfg_augmentation, seed=p.seeds[0])[0]

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            latent_modifier_enabled=enabled,
            latent_modifier_sharpness_multiplier=sharpness_multiplier,
            latent_modifier_sharpness_method=sharpness_method,
            latent_modifier_tonemap_multiplier=tonemap_multiplier,
            latent_modifier_tonemap_method=tonemap_method,
            latent_modifier_tonemap_percentile=tonemap_percentile,
            latent_modifier_contrast_multiplier=contrast_multiplier,
            latent_modifier_combat_method=combat_method,
            latent_modifier_combat_cfg_drift=combat_cfg_drift,
            latent_modifier_rescale_cfg_phi=rescale_cfg_phi,
            latent_modifier_extra_noise_type=extra_noise_type,
            latent_modifier_extra_noise_method=extra_noise_method,
            latent_modifier_extra_noise_multiplier=extra_noise_multiplier,
            latent_modifier_extra_noise_lowpass=extra_noise_lowpass,
            latent_modifier_divisive_norm_size=divisive_norm_size,
            latent_modifier_divisive_norm_multiplier=divisive_norm_multiplier,
            latent_modifier_spectral_mod_mode=spectral_mod_mode,
            latent_modifier_spectral_mod_percentile=spectral_mod_percentile,
            latent_modifier_spectral_mod_multiplier=spectral_mod_multiplier,
            latent_modifier_affect_uncond=affect_uncond,
            latent_modifier_dyn_cfg_augmentation=dyn_cfg_augmentation,
        ))

        return
