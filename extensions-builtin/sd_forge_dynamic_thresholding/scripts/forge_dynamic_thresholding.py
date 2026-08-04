import gradio as gr

from modules import scripts
from lib_dynamic_thresholding.dynthres import DynamicThresholdingNode
from modules.ui_components import InputAccordion

opDynamicThresholdingNode = DynamicThresholdingNode().patch


class DynamicThresholdingForForge(scripts.Script):
    sorting_priority = 11

    def title(self):
        return "DynamicThresholding (CFG-Fix) Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        elem = 'dynthres_'
        with InputAccordion(False, label=self.title(),elem_id=elem+'enabled') as enabled:
            mimic_scale = gr.Slider(label='Mimic Scale', minimum=0.0, maximum=100.0, step=0.5, value=7.0,elem_id=elem+'mimic_scale')
            threshold_percentile = gr.Slider(label='Threshold Percentile', minimum=0.0, maximum=1.0, step=0.01,
                                             value=1.0,elem_id=elem+'threshold_percentile')
            mimic_mode = gr.Radio(label='Mimic Mode',
                                  choices=['Constant', 'Linear Down', 'Cosine Down', 'Half Cosine Down', 'Linear Up',
                                           'Cosine Up', 'Half Cosine Up', 'Power Up', 'Power Down', 'Linear Repeating',
                                           'Cosine Repeating', 'Sawtooth'], value='Constant',elem_id=elem+'mimic_mode')
            mimic_scale_min = gr.Slider(label='Mimic Scale Min', minimum=0.0, maximum=100.0, step=0.5, value=0.0,elem_id=elem+'mimic_scale_min')
            cfg_mode = gr.Radio(label='Cfg Mode',
                                choices=['Constant', 'Linear Down', 'Cosine Down', 'Half Cosine Down', 'Linear Up',
                                         'Cosine Up', 'Half Cosine Up', 'Power Up', 'Power Down', 'Linear Repeating',
                                         'Cosine Repeating', 'Sawtooth'], value='Constant',elem_id=elem+'cfg_mode')
            cfg_scale_min = gr.Slider(label='Cfg Scale Min', minimum=0.0, maximum=100.0, step=0.5, value=0.0,elem_id=elem+'cfg_scale_min')
            sched_val = gr.Slider(label='Sched Val', minimum=0.0, maximum=100.0, step=0.01, value=1.0,elem_id=elem+'sched_val')
            separate_feature_channels = gr.Radio(label='Separate Feature Channels', choices=['enable', 'disable'],
                                                 value='enable',elem_id=elem+'separate_feature_channels')
            scaling_startpoint = gr.Radio(label='Scaling Startpoint', choices=['MEAN', 'ZERO'], value='MEAN',elem_id=elem+'scaling_startpoint')
            variability_measure = gr.Radio(label='Variability Measure', choices=['AD', 'STD'], value='AD',elem_id=elem+'variability_measure')
            interpolate_phi = gr.Slider(label='Interpolate Phi', minimum=0.0, maximum=1.0, step=0.01, value=1.0,elem_id=elem+'interpolate_phi')

        return enabled, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, \
            sched_val, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        enabled, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, \
            sched_val, separate_feature_channels, scaling_startpoint, variability_measure, \
            interpolate_phi = script_args

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = opDynamicThresholdingNode(unet, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min,
                                         cfg_mode, cfg_scale_min, sched_val, separate_feature_channels,
                                         scaling_startpoint, variability_measure, interpolate_phi)[0]

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            dynthres_enabled=enabled,
            dynthres_mimic_scale=mimic_scale,
            dynthres_threshold_percentile=threshold_percentile,
            dynthres_mimic_mode=mimic_mode,
            dynthres_mimic_scale_min=mimic_scale_min,
            dynthres_cfg_mode=cfg_mode,
            dynthres_cfg_scale_min=cfg_scale_min,
            dynthres_sched_val=sched_val,
            dynthres_separate_feature_channels=separate_feature_channels,
            dynthres_scaling_startpoint=scaling_startpoint,
            dynthres_variability_measure=variability_measure,
            dynthres_interpolate_phi=interpolate_phi,
        ))

        return
