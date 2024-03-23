import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F

from depthfm import DepthFM
from einops import rearrange
from .util import load_model
from .annotator_path import models_path
from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import preprocessor_dir, add_supported_preprocessor
from modules_forge.forge_util import resize_image_with_pad


class DepthFMDetector:
    """https://github.com/CompVis/depth-fm"""

    model_dir = os.path.join(models_path, "depth_fm")

    def __init__(self, device: torch.device):
        self.device = device
        remote_url = os.environ.get(
            "CONTROLNET_DEPTH_FM_MODEL_URL",
            "https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt",
        )
        model_path = load_model(
            "depthfm-v1.ckpt", remote_url=remote_url, model_dir=self.model_dir
        )
        self.model = DepthFM(model_path).to(device).eval()
        self.slider_resolution = PreprocessorParameter(
            label='Resolution', minimum=128, maximum=2048, value=768, step=8, visible=True)
        self.slider_1 = PreprocessorParameter(label='Num Steps', value=2, minimum=1, maximum=30, step=1, visible=True)
        self.slider_2 = PreprocessorParameter(label='Ensemble Size', value=4, minimum=1, maximum=9, step=1, visible=True)
        self.num_steps = 2
        self.ensemble_Size = 1
        #self.model.load_state_dict(torch.load(model_path))

    def __call__(self, image: np.ndarray, slider_1=None, slider_2=None, colored: bool = True) -> np.ndarray:
        self.num_steps = slider_1
        self.ensemble_Size = slider_2

        self.model.to(self.device)

        image = torch.from_numpy(image).float().to(self.device)
        image = image / 127.5 - 1
        image = rearrange(image, 'h w c -> 1 c h w')
        @torch.no_grad()
        def predict_depth(model, image, num_steps, ensemble_Size):
            return model.predict_depth(image, num_steps, ensemble_Size)
        depth = predict_depth(self.model, image,self.num_steps,self.ensemble_Size)
        depth = depth * 255.0
        depth = depth.squeeze(0).squeeze(0).cpu().numpy()
        depth = 255 - depth.astype(np.uint8)
        if colored:
            return cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        else:
            return depth

    def unload_model(self):
        self.model.to("cpu")