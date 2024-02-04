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

        self.diffusers_patcher = DiffusersModelPatcher(
            pipeline_class=MarigoldPipeline,
            pretrained_path="Bingxin/Marigold",
            enable_xformers=False,
            noise_scheduler_type='DDIMScheduler')

        return

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        input_image, remove_pad = resize_image_with_pad(input_image, resolution)

        self.load_model()

        H, W, C = input_image.shape

        self.diffusers_patcher.prepare_memory_before_sampling(
            batchsize=1, latent_width=W // 8, latent_height=H // 8
        )

        with torch.no_grad():
            img = numpy_to_pytorch(input_image).movedim(-1, 1)
            img = self.diffusers_patcher.move_tensor_to_current_device(img)

            img = img * 2.0 - 1.0
            depth = self.diffusers_patcher.pipeline(img, num_inference_steps=20, show_pbar=False)
            depth = 0.5 - depth * 0.5
            depth = depth.movedim(1, -1)[0].cpu().numpy()
            depth_image = HWC3((depth * 255.0).clip(0, 255).astype(np.uint8))

        return remove_pad(depth_image)


add_supported_preprocessor(PreprocessorMarigold())
