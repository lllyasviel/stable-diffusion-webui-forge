import contextlib
import torch
import ldm_patched.modules.model_management as model_management


def has_xpu() -> bool:
    return model_management.xpu_available


def has_mps() -> bool:
    return model_management.mps_mode()


def cuda_no_autocast(device_id=None) -> bool:
    return False


def get_cuda_device_id():
    return model_management.get_torch_device().index


def get_cuda_device_string():
    return str(model_management.get_torch_device())


def get_optimal_device_name():
    return model_management.get_torch_device().type


def get_optimal_device():
    return model_management.get_torch_device()


def get_device_for(task):
    return model_management.get_torch_device()


def torch_gc():
    model_management.soft_empty_cache()


def torch_npu_set_device():
    return


def enable_tf32():
    return


cpu: torch.device = torch.device("cpu")
fp8: bool = False
device: torch.device = model_management.get_torch_device()
device_interrogate: torch.device = model_management.text_encoder_device()  # for backward compatibility, not used now
device_gfpgan: torch.device = model_management.get_torch_device()  # will be managed by memory management system
device_esrgan: torch.device = model_management.get_torch_device()  # will be managed by memory management system
device_codeformer: torch.device = model_management.get_torch_device()  # will be managed by memory management system
dtype: torch.dtype = model_management.unet_dtype()
dtype_vae: torch.dtype = model_management.vae_dtype()
dtype_unet: torch.dtype = model_management.unet_dtype()
dtype_inference: torch.dtype = model_management.unet_dtype()
unet_needs_upcast = False


def cond_cast_unet(input):
    return input


def cond_cast_float(input):
    return input


nv_rng = None
patch_module_list = []


def manual_cast_forward(target_dtype):
    return


@contextlib.contextmanager
def manual_cast(target_dtype):
    return


def autocast(disable=False):
    return contextlib.nullcontext()


def without_autocast(disable=False):
    return contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    return


def first_time_calculation():
    return
