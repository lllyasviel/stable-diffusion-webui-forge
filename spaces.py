from modules_forge.initialization import initialize_forge

initialize_forge()

import os
import torch
import inspect
import functools
import gradio.oauth
import gradio.routes

from backend import memory_management
from diffusers.models import modeling_utils as diffusers_modeling_utils
from transformers import modeling_utils as transformers_modeling_utils
from backend.attention import AttentionProcessorForge
from starlette.requests import Request


_original_init = Request.__init__


def patched_init(self, scope, receive=None, send=None):
    if 'session' not in scope:
        scope['session'] = dict()
    _original_init(self, scope, receive, send)
    return


Request.__init__ = patched_init
gradio.oauth.attach_oauth = lambda x: None
gradio.routes.attach_oauth = lambda x: None

module_in_gpu: torch.nn.Module = None
gpu = memory_management.get_torch_device()
cpu = torch.device('cpu')

diffusers_modeling_utils.get_parameter_device = lambda *args, **kwargs: gpu
transformers_modeling_utils.get_parameter_device = lambda *args, **kwargs: gpu


def unload_module():
    global module_in_gpu

    if module_in_gpu is None:
        return

    print(f'Moved module to CPU: {type(module_in_gpu).__name__}')
    module_in_gpu.to(cpu)
    module_in_gpu = None
    memory_management.soft_empty_cache()
    return


def load_module(m):
    global module_in_gpu

    if module_in_gpu == m:
        return

    unload_module()
    module_in_gpu = m
    module_in_gpu.to(gpu)
    print(f'Moved module to GPU: {type(module_in_gpu).__name__}')
    return


class GPUObject:
    def __init__(self):
        self.module_list = []

    def __enter__(self):
        self.original_init = torch.nn.Module.__init__
        self.original_to = torch.nn.Module.to

        def patched_init(module, *args, **kwargs):
            self.module_list.append(module)
            return self.original_init(module, *args, **kwargs)

        def patched_to(module, *args, **kwargs):
            self.module_list.append(module)
            return self.original_to(module, *args, **kwargs)

        torch.nn.Module.__init__ = patched_init
        torch.nn.Module.to = patched_to
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.nn.Module.__init__ = self.original_init
        torch.nn.Module.to = self.original_to
        self.module_list = set(self.module_list)
        self.to(device=torch.device('cpu'))
        memory_management.soft_empty_cache()
        return

    def to(self, device):
        for module in self.module_list:
            module.to(device)
        print(f'Forge Space: Moved {len(self.module_list)} Modules to {device}')
        return self

    def gpu(self):
        self.to(device=gpu)
        return self


def GPU(gpu_objects=None, manual_load=False):
    gpu_objects = gpu_objects or []

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print("Entering Forge Space GPU ...")
            memory_management.unload_all_models()
            if not manual_load:
                for o in gpu_objects:
                    o.gpu()
            result = func(*args, **kwargs)
            print("Cleaning Forge Space GPU ...")
            unload_module()
            for o in gpu_objects:
                o.to(device=torch.device('cpu'))
            memory_management.soft_empty_cache()
            return result
        return wrapper
    return decorator


def convert_root_path():
    frame = inspect.currentframe().f_back
    caller_file = frame.f_code.co_filename
    caller_file = os.path.abspath(caller_file)
    result = os.path.join(os.path.dirname(caller_file), 'huggingface_space_mirror')
    return result + '/'


def automatically_move_to_gpu_when_forward(m: torch.nn.Module, target_model: torch.nn.Module = None):
    if target_model is None:
        target_model = m

    def patch_method(method_name):
        if not hasattr(m, method_name):
            return

        if not hasattr(m, 'forge_space_hooked_names'):
            m.forge_space_hooked_names = []

        if method_name in m.forge_space_hooked_names:
            return

        print(f'Automatic hook: {type(m).__name__}.{method_name}')

        original_method = getattr(m, method_name)

        def patched_method(*args, **kwargs):
            load_module(target_model)
            return original_method(*args, **kwargs)

        setattr(m, method_name, patched_method)

        m.forge_space_hooked_names.append(method_name)
        return

    for method_name in ['forward', 'encode', 'decode']:
        patch_method(method_name)

    return


def automatically_move_pipeline_components(pipe):
    for attr_name in dir(pipe):
        attr_value = getattr(pipe, attr_name, None)
        if isinstance(attr_value, torch.nn.Module):
            automatically_move_to_gpu_when_forward(attr_value)
    return


def change_attention_from_diffusers_to_forge(m):
    m.set_attn_processor(AttentionProcessorForge())
    return
