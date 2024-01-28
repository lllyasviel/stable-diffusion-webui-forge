import diffusers
import torch
import ldm_patched.modules.ops as ops
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules import model_management
from modules_forge.ops import use_patched_ops
from transformers import modeling_utils


class DiffusersPatcher:
    def __init__(self, pipeline_class, dtype=torch.float16, *args, **kwargs):
        load_device = model_management.get_torch_device()
        offload_device = torch.device("cpu")

        if not model_management.should_use_fp16(device=load_device):
            dtype = torch.float32

        self.dtype = dtype

        with use_patched_ops(ops.manual_cast):
            with modeling_utils.no_init_weights():
                self.pipeline = pipeline_class.from_pretrained(*args, **kwargs)

        self.pipeline = self.pipeline.to(device=offload_device, dtype=dtype)
        self.pipeline.eval()

        self.patcher = ModelPatcher(
            model=self.pipeline,
            load_device=load_device,
            offload_device=offload_device)

    def prepare_memory_before_sampling(self, batchsize, latent_width, latent_height):
        area = 2 * batchsize * latent_width * latent_height
        inference_memory = (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)
        model_management.load_models_gpu(
            models=[self.pipeline],
            memory_required=inference_memory
        )