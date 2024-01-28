import cv2
import os

from modules.paths import models_path


controlnet_dir = os.path.join(models_path, 'ControlNet')
os.makedirs(controlnet_dir, exist_ok=True)

preprocessor_dir = os.path.join(models_path, 'ControlNetPreprocessor')
os.makedirs(preprocessor_dir, exist_ok=True)

shared_preprocessors = {}


class PreprocessorParameter:
    def __init__(self, minimum=0.0, maximum=1.0, step=0.01, label='Parameter 1', value=0.5, visible=False):
        self.gradio_update_kwargs = dict(
            minimum=minimum, maximum=maximum, step=step, label=label, value=value, visible=visible
        )


class PreprocessorBase:
    def __init__(self):
        self.name = 'PreprocessorBase'
        self.slider_1 = PreprocessorParameter()
        self.slider_2 = PreprocessorParameter()
        self.slider_3 = PreprocessorParameter()

    def __call__(self, input_image, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        return input_image


class PreprocessorNone(PreprocessorBase):
    def __init__(self):
        super().__init__()
        self.name = 'None'

    def __call__(self, input_image, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        return input_image


shared_preprocessors['none'] = PreprocessorNone()


class PreprocessorCanny(PreprocessorBase):
    def __init__(self):
        super().__init__()
        self.name = 'canny'
        self.slider_1 = PreprocessorParameter(minimum=0, maximum=256, step=1, value=100, label='Low Threshold', visible=True)
        self.slider_2 = PreprocessorParameter(minimum=0, maximum=256, step=1, value=200, label='High Threshold', visible=True)

    def __call__(self, input_image, slider_1=100, slider_2=200, slider_3=None, **kwargs):
        canny_image = cv2.cvtColor(cv2.Canny(input_image, int(slider_1), int(slider_2)), cv2.COLOR_GRAY2RGB)
        return canny_image


shared_preprocessors['canny'] = PreprocessorCanny()
