from backend import memory_management
from backend.patcher.base import ModelPatcher
from backend.nn.base import ModuleDict, ObjectDict


class JointTextEncoder(ModuleDict):
    pass


class CLIP:
    def __init__(self, model_dict={}, tokenizer_dict={}, no_init=False):
        if no_init:
            return

        load_device = memory_management.text_encoder_device()
        offload_device = memory_management.text_encoder_offload_device()

        self.cond_stage_model = JointTextEncoder(model_dict)
        self.tokenizer = ObjectDict(tokenizer_dict)
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)
