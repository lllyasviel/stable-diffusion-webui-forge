import cv2
import numpy as np

from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor


class PreprocessorRecolor(Preprocessor):
    def __init__(self, name, use_intensity):
        super().__init__()
        self.name = name
        self.use_intensity = False
        self.tags = ['Recolor']
        self.model_filename_filters = ['color', 'recolor', 'grey', 'gray']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.slider_1 = PreprocessorParameter(
            visible=True,
            label="Gamma Correction",
            value=1.0,
            minimum=0.1,
            maximum=2.0,
            step=0.001
        )
        self.current_cond = None

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        gamma = slider_1

        if self.use_intensity:
            result = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
            result = result[:, :, 2].astype(np.float32) / 255.0
        else:
            result = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
            result = result[:, :, 0].astype(np.float32) / 255.0

        result = result ** gamma
        result = (result * 255.0).clip(0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        self.current_cond = cond
        return cond, mask

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        a1111_batch_result = args[0]
        new_results = []

        for img in a1111_batch_result.images:
            new_mean = self.current_cond[0].mean(dim=0, keepdim=True)
            img = img - img.mean(dim=0, keepdim=True) + new_mean
            img = img.clip(0, 1)
            new_results.append(img)

        a1111_batch_result.images = new_results
        return


add_supported_preprocessor(PreprocessorRecolor(
    name="recolor_intensity",
    use_intensity=True
))

add_supported_preprocessor(PreprocessorRecolor(
    name="recolor_luminance",
    use_intensity=False
))
