import gradio as gr
from modules import scripts


DEFAULT_UNITS = 1
MAX_UNITS = 16


class ControlNetForge(scripts.Script):
    def title(self):
        return "ControlNet Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def build_unit_ui(self):
        return

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            unit_count = gr.Slider(label='ControlNet Units', minimum=0, maximum=MAX_UNITS, step=1, value=DEFAULT_UNITS)
            unit_accordions = []
            for i in range(16):
                with gr.Accordion(open=True, label=f'ControlNet Unit {i+1}',
                                  visible=i < DEFAULT_UNITS) as unit_accordion:
                    self.build_unit_ui()
                unit_accordions.append(unit_accordion)

            unit_count.change(lambda c: [gr.update(visible=i < c) for i in range(MAX_UNITS)],
                              inputs=unit_count, outputs=unit_accordions, show_progress=False, queue=False)

        return [unit_count]
