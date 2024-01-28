from modules_forge.shared import Preprocessor, preprocessor_dir, load_file_from_url, add_preprocessor

import os
import types
import torch
import numpy as np

from einops import rearrange
from annotator.normalbae.models.NNET import NNET
import torchvision.transforms as transforms


def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model


class PreprocessorNormalBae(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'normalbae'
        self.tag = 'NormalMap'

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_model(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt"
        modelpath = os.path.join(preprocessor_dir, "scannet.pt")
        if not os.path.exists(modelpath):
            load_file_from_url(remote_model_path, model_dir=preprocessor_dir)
        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = load_checkpoint(modelpath, model)
        model.eval()
        self.setup_model_patcher(model)

    def __call__(self, input_image):
        if self.model_patcher is None:
            self.load_model()

        self.load_models_gpu()

        assert input_image.ndim == 3
        image_normal = input_image
        with torch.no_grad():
            image_normal = self.send_tensor_to_model_device(torch.from_numpy(image_normal))
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            normal = self.model_patcher.model(image_normal)
            normal = normal[0][-1][:, :3]
            # d = torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
            # d = torch.maximum(d, torch.ones_like(d) * 1e-5)
            # normal /= d
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

            return normal_image


add_preprocessor(PreprocessorNormalBae)
