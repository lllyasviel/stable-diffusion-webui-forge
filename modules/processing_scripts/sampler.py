import gradio as gr

from modules import scripts, sd_samplers, sd_schedulers, shared
from modules.infotext_utils import PasteField
from modules.ui_components import FormRow, FormGroup


class ScriptSampler(scripts.ScriptBuiltinUI):
    section = "sampler"

    def __init__(self):
        self.steps = None
        self.sampler_name = None
        self.scheduler = None

    def title(self):
        return "Sampler"

    def ui(self, is_img2img):
        sampler_names = [x.name for x in sd_samplers.visible_samplers()]
        scheduler_names = [x.label for x in sd_schedulers.schedulers]

        if shared.opts.samplers_in_dropdown:
            with FormRow(elem_id=f"sampler_selection_{self.tabname}"):
                self.sampler_name = gr.Dropdown(label='Sampling method', elem_id=f"{self.tabname}_sampling", choices=sampler_names, value=sampler_names[0])
                self.scheduler = gr.Dropdown(label='Schedule type', elem_id=f"{self.tabname}_scheduler", choices=scheduler_names, value=scheduler_names[0])
                self.steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{self.tabname}_steps", label="Sampling steps", value=20)
        else:
            with FormGroup(elem_id=f"sampler_selection_{self.tabname}"):
                self.steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{self.tabname}_steps", label="Sampling steps", value=20)
                self.sampler_name = gr.Radio(label='Sampling method', elem_id=f"{self.tabname}_sampling", choices=sampler_names, value=sampler_names[0])
                self.scheduler = gr.Dropdown(label='Schedule type', elem_id=f"{self.tabname}_scheduler", choices=scheduler_names, value=scheduler_names[0])

        self.infotext_fields = [
            PasteField(self.steps, "Steps", api="steps"),
            PasteField(self.sampler_name, sd_samplers.get_sampler_from_infotext, api="sampler_name"),
            PasteField(self.scheduler, sd_samplers.get_scheduler_from_infotext, api="scheduler"),
        ]

        shared.options_templates.update(shared.options_section(('ui_sd', "UI defaults 'sd'", "ui"), {
            "sd_t2i_sampler":     shared.OptionInfo('Euler a',      "txt2img sampler",      gr.Dropdown, {"choices": sampler_names}),
            "sd_t2i_scheduler":   shared.OptionInfo('Automatic',    "txt2img scheduler",    gr.Dropdown, {"choices": scheduler_names}),
            "sd_i2i_sampler":     shared.OptionInfo('Euler a',      "img2img sampler",      gr.Dropdown, {"choices": sampler_names}),
            "sd_i2i_scheduler":   shared.OptionInfo('Automatic',    "img2img scheduler",    gr.Dropdown, {"choices": scheduler_names}),
        }))
        shared.options_templates.update(shared.options_section(('ui_xl', "UI defaults 'xl'", "ui"), {
            "xl_t2i_sampler":     shared.OptionInfo('DPM++ 2M SDE', "txt2img sampler",      gr.Dropdown, {"choices": sampler_names}),
            "xl_t2i_scheduler":   shared.OptionInfo('Karras',       "txt2img scheduler",    gr.Dropdown, {"choices": scheduler_names}),
            "xl_i2i_sampler":     shared.OptionInfo('DPM++ 2M SDE', "img2img sampler",      gr.Dropdown, {"choices": sampler_names}),
            "xl_i2i_scheduler":   shared.OptionInfo('Karras',       "img2img scheduler",    gr.Dropdown, {"choices": scheduler_names}),
        }))
        shared.options_templates.update(shared.options_section(('ui_flux', "UI defaults 'flux'", "ui"), {
            "flux_t2i_sampler":   shared.OptionInfo('Euler',        "txt2img sampler",      gr.Dropdown, {"choices": sampler_names}),
            "flux_t2i_scheduler": shared.OptionInfo('Simple',       "txt2img scheduler",    gr.Dropdown, {"choices": scheduler_names}),
            "flux_i2i_sampler":   shared.OptionInfo('Euler',        "img2img sampler",      gr.Dropdown, {"choices": sampler_names}),
            "flux_i2i_scheduler": shared.OptionInfo('Simple',       "img2img scheduler",    gr.Dropdown, {"choices": scheduler_names}),
        }))

        return self.steps, self.sampler_name, self.scheduler

    def setup(self, p, steps, sampler_name, scheduler):
        p.steps = steps
        p.sampler_name = sampler_name
        p.scheduler = scheduler
