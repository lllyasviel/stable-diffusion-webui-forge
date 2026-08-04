import gradio as gr
from modules import scripts
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

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
        elem = 'latent_modifier_'
        with InputAccordion(False, label=self.title(),elem_id=elem+'enabled') as enabled:
            sharpness_multiplier = gr.Slider(label='Sharpness Multiplier', minimum=-100.0, maximum=100.0, step=0.1,
                                             value=0.0,elem_id=elem+'sharpness_multiplier')
            sharpness_method = gr.Radio(label='Sharpness Method',
                                        choices=['anisotropic', 'joint-anisotropic', 'gaussian', 'cas'],
                                        value='anisotropic',elem_id=elem+'sharpness_method')
            tonemap_multiplier = gr.Slider(label='Tonemap Multiplier', minimum=0.0, maximum=100.0, step=0.01, value=0.0,elem_id=elem+'tonemap_multiplier')
            tonemap_method = gr.Radio(label='Tonemap Method',
                                      choices=['reinhard', 'reinhard_perchannel', 'arctan', 'quantile', 'gated',
                                               'cfg-mimic', 'spatial-norm'], value='reinhard',elem_id=elem+'tonemap_method')
            tonemap_percentile = gr.Slider(label='Tonemap Percentile', minimum=0.0, maximum=100.0, step=0.005,
                                           value=100.0,elem_id=elem+'tonemap_percentile')
            contrast_multiplier = gr.Slider(label='Contrast Multiplier', minimum=-100.0, maximum=100.0, step=0.1,
                                            value=0.0,elem_id=elem+'contrast_multiplier')
            combat_method = gr.Radio(label='Combat Method',
                                     choices=['subtract', 'subtract_channels', 'subtract_median', 'sharpen'],
                                     value='subtract',elem_id=elem+'combat_method')
            combat_cfg_drift = gr.Slider(label='Combat Cfg Drift', minimum=-10.0, maximum=10.0, step=0.01, value=0.0,elem_id=elem+'combat_cfg_drift')
            rescale_cfg_phi = gr.Slider(label='Rescale Cfg Phi', minimum=-10.0, maximum=10.0, step=0.01, value=0.0,elem_id=elem+'rescale_cfg_phi')
            extra_noise_type = gr.Radio(label='Extra Noise Type',
                                        choices=['gaussian', 'uniform', 'perlin', 'pink', 'green', 'pyramid'],
                                        value='gaussian',elem_id=elem+'extra_noise_type')
            extra_noise_method = gr.Radio(label='Extra Noise Method',
                                          choices=['add', 'add_scaled', 'speckle', 'cads', 'cads_rescaled',
                                                   'cads_speckle', 'cads_speckle_rescaled'], value='add',elem_id=elem+'extra_noise_method')
            extra_noise_multiplier = gr.Slider(label='Extra Noise Multiplier', minimum=0.0, maximum=100.0, step=0.1,
                                               value=0.0,elem_id=elem+'extra_noise_multiplier')
            extra_noise_lowpass = gr.Slider(label='Extra Noise Lowpass', minimum=0, maximum=1000, step=1, value=100,elem_id=elem+'extra_noise_lowpass')
            divisive_norm_size = gr.Slider(label='Divisive Norm Size', minimum=1, maximum=255, step=1, value=127,elem_id=elem+'divisive_norm_size')
            divisive_norm_multiplier = gr.Slider(label='Divisive Norm Multiplier', minimum=0.0, maximum=1.0, step=0.01,
                                                 value=0.0,elem_id=elem+'divisive_norm_multiplier')
            spectral_mod_mode = gr.Radio(label='Spectral Mod Mode', choices=['hard_clamp', 'soft_clamp'],
                                         value='hard_clamp',elem_id=elem+'spectral_mod_mode')
            spectral_mod_percentile = gr.Slider(label='Spectral Mod Percentile', minimum=0.0, maximum=50.0, step=0.01,
                                                value=5.0,elem_id=elem+'spectral_mod_percentile')
            spectral_mod_multiplier = gr.Slider(label='Spectral Mod Multiplier', minimum=-15.0, maximum=15.0, step=0.01,
                                                value=0.0,elem_id=elem+'spectral_mod_multiplier')
            affect_uncond = gr.Radio(label='Affect Uncond', choices=['None', 'Sharpness'], value='None',elem_id=elem+'affect_uncond')
            dyn_cfg_augmentation = gr.Radio(label='Dyn Cfg Augmentation',
                                            choices=['None', 'dyncfg-halfcosine', 'dyncfg-halfcosine-mimic'],
                                            value='None',elem_id=elem+'dyn_cfg_augmentation')
        
        self.infotext_fields = [
            PasteField(enabled, "latent_modifier_enabled", api="latent_modifier_enabled"),
            PasteField(sharpness_multiplier, "latent_modifier_sharpness_multiplier", api="latent_modifier_sharpness_multiplier"),
            PasteField(sharpness_method, "latent_modifier_sharpness_method", api="latent_modifier_sharpness_method"),
            PasteField(tonemap_multiplier, "latent_modifier_tonemap_multiplier", api="latent_modifier_tonemap_multiplier"),
            PasteField(tonemap_method, "latent_modifier_tonemap_method", api="latent_modifier_tonemap_method"),
            PasteField(tonemap_percentile, "latent_modifier_tonemap_percentile", api="latent_modifier_tonemap_percentile"),
            PasteField(contrast_multiplier, "latent_modifier_contrast_multiplier", api="latent_modifier_contrast_multiplier"),
            PasteField(combat_method, "latent_modifier_combat_method", api="latent_modifier_combat_method"),
            PasteField(combat_cfg_drift, "latent_modifier_combat_cfg_drift", api="latent_modifier_combat_cfg_drift"),
            PasteField(rescale_cfg_phi, "latent_modifier_rescale_cfg_phi", api="latent_modifier_rescale_cfg_phi"),
            PasteField(extra_noise_type, "latent_modifier_extra_noise_type", api="latent_modifier_extra_noise_type"),
            PasteField(extra_noise_method, "latent_modifier_extra_noise_method", api="latent_modifier_extra_noise_method"),
            PasteField(extra_noise_multiplier, "latent_modifier_extra_noise_multiplier", api="latent_modifier_extra_noise_multiplier"),
            PasteField(extra_noise_lowpass, "latent_modifier_extra_noise_lowpass", api="latent_modifier_extra_noise_lowpass"),
            PasteField(divisive_norm_size, "latent_modifier_divisive_norm_size", api="latent_modifier_divisive_norm_size"),
            PasteField(divisive_norm_multiplier, "latent_modifier_divisive_norm_multiplier", api="latent_modifier_divisive_norm_multiplier"),
            PasteField(spectral_mod_mode, "latent_modifier_spectral_mod_mode", api="latent_modifier_spectral_mod_mode"),
            PasteField(spectral_mod_percentile, "latent_modifier_spectral_mod_percentile", api="latent_modifier_spectral_mod_percentile"),
            PasteField(spectral_mod_multiplier, "latent_modifier_spectral_mod_multiplier", api="latent_modifier_spectral_mod_multiplier"),
            PasteField(affect_uncond, "latent_modifier_affect_uncond", api="latent_modifier_affect_uncond"),
            PasteField(dyn_cfg_augmentation, "latent_modifier_dyn_cfg_augmentation", api="latent_modifier_dyn_cfg_augmentation"),
        ]
        self.paste_field_names = []
        for field in self.infotext_fields:
            self.paste_field_names.append(field.api)
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
