from modules_forge.initialization import initialize_forge

initialize_forge()

import os
import torch
import inspect

from backend import memory_management


gpu = memory_management.get_torch_device()


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
        def wrapper(*args, **kwargs):
            print("Entering Forge Space GPU ...")
            memory_management.unload_all_models()
            if not manual_load:
                for o in gpu_objects:
                    o.gpu()
            result = func(*args, **kwargs)
            print("Cleaning Forge Space GPU ...")
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
