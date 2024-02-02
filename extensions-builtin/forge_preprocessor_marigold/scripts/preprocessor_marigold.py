from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import preprocessor_dir, add_supported_preprocessor
from modules_forge.forge_util import resize_image_with_pad


import os
import torch
import numpy as np

from marigold.model.marigold_pipeline import MarigoldPipeline
from huggingface_hub import snapshot_download
from modules_forge.diffusers_patcher import DiffusersModelPatcher
from modules_forge.forge_util import numpy_to_pytorch, HWC3
from ldm_patched.modules import model_management


class PreprocessorMarigold(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'depth_marigold'
        self.tags = ['Depth']
        self.model_filename_filters = ['depth']
        self.slider_resolution = PreprocessorParameter(
            label='Resolution', minimum=128, maximum=2048, value=768, step=8, visible=True)
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.slider_3 = PreprocessorParameter(visible=False)
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 100  # higher goes to top in the list
        self.diffusers_patcher = None

    def load_model(self):
        if self.model_patcher is not None:
            return

        checkpoint_path = os.path.join(preprocessor_dir, 'marigold')

        if not os.path.exists(checkpoint_path):
            snapshot_download(repo_id="Bingxin/Marigold",
                              ignore_patterns=["*.bin"],
                              local_dir=checkpoint_path,
                              local_dir_use_symlinks=False)

        self.diffusers_patcher = DiffusersModelPatcher(
            pipeline_class=MarigoldPipeline,
            pretrained_path=checkpoint_path,
            enable_xformers=False,
            noise_scheduler_type='DDIMScheduler')

        return

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        input_image, remove_pad = resize_image_with_pad(input_image, resolution)

        self.load_model()

        model_management.load_models_gpu([self.diffusers_patcher.patcher])

        with torch.no_grad():
            img = numpy_to_pytorch(input_image).movedim(-1, 1).to(
                device=self.diffusers_patcher.patcher.current_device,
                dtype=self.diffusers_patcher.dtype)

            img = img * 2.0 - 1.0
            depth = self.diffusers_patcher.patcher.model(img, num_inference_steps=20, show_pbar=False)
            depth = depth * 0.5 + 0.5
            depth = depth.movedim(1, -1)[0].cpu().numpy()
            depth_image = HWC3((depth * 255.0).clip(0, 255).astype(np.uint8))

        return remove_pad(depth_image)


add_supported_preprocessor(PreprocessorMarigold())
