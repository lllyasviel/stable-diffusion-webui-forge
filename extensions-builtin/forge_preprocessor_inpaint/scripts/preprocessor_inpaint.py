from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.forge_util import numpy_to_pytorch


class PreprocessorInpaint(Preprocessor):
    def __init__(self, name, use_inpaint_sampler=False, use_lama=False):
        super().__init__()
        self.name = name
        self.use_inpaint_sampler = use_inpaint_sampler
        self.use_lama = use_lama
        self.tags = ['Inpaint']
        self.model_filename_filters = ['inpaint']
        self.slider_resolution = PreprocessorParameter(visible=False)

    def process_before_every_sampling(self, process, cond, *args, **kwargs):
        return

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        return


add_supported_preprocessor(PreprocessorInpaint(
    name='inpaint_global_harmonious',
    use_inpaint_sampler=False,
    use_lama=False
))

add_supported_preprocessor(PreprocessorInpaint(
    name='inpaint_only',
    use_inpaint_sampler=False,
    use_lama=False
))

add_supported_preprocessor(PreprocessorInpaint(
    name='inpaint_only+lama',
    use_inpaint_sampler=False,
    use_lama=False
))
