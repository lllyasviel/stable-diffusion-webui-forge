import torch

from backend import memory_management
from backend.patcher.base import ModelPatcher


class JointTokenizer:
    def __init__(self, huggingface_components):
        self.clip_l = huggingface_components.get('tokenizer', None)
        self.clip_g = huggingface_components.get('tokenizer_2', None)


class JointCLIPTextEncoder(torch.nn.Module):
    def __init__(self, huggingface_components):
        super().__init__()
        self.clip_l = huggingface_components.get('text_encoder', None)
        self.clip_g = huggingface_components.get('text_encoder_2', None)


class CLIP:
    def __init__(self, huggingface_components=None, no_init=False):
        if no_init:
            return

        load_device = memory_management.text_encoder_device()
        offload_device = memory_management.text_encoder_offload_device()
        text_encoder_dtype = memory_management.text_encoder_dtype(load_device)

        self.cond_stage_model = JointCLIPTextEncoder(huggingface_components)
        self.tokenizer = JointTokenizer(huggingface_components)
        self.cond_stage_model.to(dtype=text_encoder_dtype, device=offload_device)
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)
