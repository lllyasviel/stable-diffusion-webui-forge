from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import preprocessor_dir, add_supported_preprocessor
from modules_forge.forge_util import resize_image_with_pad
from modules.modelloader import load_file_from_url
from ldm_patched.modules import model_management

import torch
import numpy as np

from einops import rearrange
from annotator.normaldsine.models.dsine import DSINE
from annotator.normaldsine.utils.utils import load_checkpoint, get_intrins_from_fov, get_pad
from torchvision import transforms


class PreprocessorNormalDsine(Preprocessor):
    def __init__(self):
        super().__init__()
        self.device = model_management.get_torch_device()
        self.name = 'normaldsine'
        self.tags = ['NormalMap']
        self.model_filename_filters = ['normal']
        self.slider_resolution = PreprocessorParameter(
            label='Resolution', minimum=128, maximum=2048, value=512, step=8, visible=True)
        self.slider_1 = PreprocessorParameter(label='fov', value=60.0, minimum=0.0, maximum=365.0, step=1.0, visible=True)
        self.slider_2 = PreprocessorParameter(label='iterations', value=5, minimum=1, maximum=20, step=1, visible=True)
        self.fov = 60.0
        self.iterations = 5
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 40  # higher goes to top in the list

    def load_model(self):
        if self.model is not None:
            return

        model_path = load_file_from_url(
            "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/dsine.pt",
            model_dir=preprocessor_dir)

        model = DSINE()
        model = load_checkpoint(model_path, model)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model.eval()
        self.model = model
        
    def to(self, device):
        self.model.to(device)
        self.model.pixel_coords = self.model.pixel_coords.to(device)
        self.device = device
        return self

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        fov = slider_1
        iterations = slider_2

        orig_H, orig_W = input_image.shape[:2]
        l, r, t, b = get_pad(orig_H, orig_W)
        input_image, remove_pad = resize_image_with_pad(input_image, resolution)

        self.load_model()
        self.model.num_iter = iterations

        assert input_image.ndim == 3
        image_normal = input_image

        with torch.no_grad():
            image_normal = torch.from_numpy(input_image).float().to(self.device)
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)
            
            intrins = get_intrins_from_fov(new_fov=fov, H=orig_H, W=orig_W, device=self.device).unsqueeze(0)
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t

            normal = self.model(image_normal, intrins)
            normal = normal[-1][0]
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        return remove_pad(normal_image)


add_supported_preprocessor(PreprocessorNormalDsine())
