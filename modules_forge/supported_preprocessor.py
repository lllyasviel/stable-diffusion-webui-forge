import cv2
import torch

from modules_forge.shared import add_supported_preprocessor
from ldm_patched.modules import model_management
from ldm_patched.modules.model_patcher import ModelPatcher
from modules_forge.forge_util import resize_image_with_pad


class PreprocessorParameter:
    def __init__(self, minimum=0.0, maximum=1.0, step=0.01, label='Parameter 1', value=0.5, visible=False, **kwargs):
        self.gradio_update_kwargs = dict(
            minimum=minimum, maximum=maximum, step=step, label=label, value=value, visible=visible, **kwargs
        )


class Preprocessor:
    def __init__(self):
        self.name = 'PreprocessorBase'
        self.tags = []
        self.model_filename_filters = []
        self.slider_resolution = PreprocessorParameter(label='Resolution', minimum=128, maximum=2048, value=512, step=8, visible=True)
        self.slider_1 = PreprocessorParameter()
        self.slider_2 = PreprocessorParameter()
        self.slider_3 = PreprocessorParameter()
        self.model_patcher: ModelPatcher = None
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 0  # higher goes to top in the list
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = True

    def setup_model_patcher(self, model, load_device=None, offload_device=None, dtype=torch.float32, **kwargs):
        if load_device is None:
            load_device = model_management.get_torch_device()

        if offload_device is None:
            offload_device = torch.device('cpu')

        if not model_management.should_use_fp16(load_device):
            dtype = torch.float32

        model.eval()
        model = model.to(device=offload_device, dtype=dtype)

        self.model_patcher = ModelPatcher(model=model, load_device=load_device, offload_device=offload_device, **kwargs)
        self.model_patcher.dtype = dtype
        return self.model_patcher

    def move_all_model_patchers_to_gpu(self):
        model_management.load_models_gpu([self.model_patcher])
        return

    def send_tensor_to_model_device(self, x):
        return x.to(device=self.model_patcher.current_device, dtype=self.model_patcher.dtype)

    def process_after_running_preprocessors(self, process, params, *args, **kwargs):
        return

    def process_before_every_sampling(self, process, cond, *args, **kwargs):
        return

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        return input_image


class PreprocessorNone(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'None'
        self.sorting_priority = 10


class PreprocessorCanny(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'canny'
        self.tags = ['Canny']
        self.model_filename_filters = ['canny']
        self.slider_1 = PreprocessorParameter(minimum=0, maximum=256, step=1, value=100, label='Low Threshold', visible=True)
        self.slider_2 = PreprocessorParameter(minimum=0, maximum=256, step=1, value=200, label='High Threshold', visible=True)
        self.sorting_priority = 100

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        input_image, remove_pad = resize_image_with_pad(input_image, resolution)
        canny_image = cv2.cvtColor(cv2.Canny(input_image, int(slider_1), int(slider_2)), cv2.COLOR_GRAY2RGB)
        return remove_pad(canny_image)


add_supported_preprocessor(PreprocessorNone())
add_supported_preprocessor(PreprocessorCanny())