from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.forge_util import numpy_to_pytorch


class PreprocessorInpaint(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'inpaint_global_harmonious'
        self.tags = ['Inpaint']
        self.model_filename_filters = ['inpaint']
        self.slider_resolution = PreprocessorParameter(visible=False)


class PreprocessorInpaintOnly(PreprocessorInpaint):
    def __init__(self):
        super().__init__()
        self.name = 'inpaint_global_harmonious'
        self.tags = ['Inpaint']
        self.model_filename_filters = ['inpaint']
        self.slider_resolution = PreprocessorParameter(visible=False)

    def process_before_every_sampling(self, process, cond, *args, **kwargs):
        return

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        return


class PreprocessorInpaintLama(PreprocessorInpaintOnly):
    def __init__(self):
        super().__init__()
        self.name = 'inpaint_global_harmonious'
        self.tags = ['Inpaint']
        self.model_filename_filters = ['inpaint']
        self.slider_resolution = PreprocessorParameter(visible=False)


add_supported_preprocessor(PreprocessorInpaint())

add_supported_preprocessor(PreprocessorInpaintOnly())

add_supported_preprocessor(PreprocessorInpaintLama())
