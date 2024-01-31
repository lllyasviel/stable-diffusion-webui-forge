import time
import torch
import contextlib
from ldm_patched.modules import model_management
from ldm_patched.modules.ops import use_patched_ops


@contextlib.contextmanager
def automatic_memory_management():
    model_management.free_memory(
        memory_required=3 * 1024 * 1024 * 1024,
        device=model_management.get_torch_device()
    )

    module_list = []

    original_init = torch.nn.Module.__init__
    original_to = torch.nn.Module.to

    def patched_init(self, *args, **kwargs):
        module_list.append(self)
        return original_init(self, *args, **kwargs)

    def patched_to(self, *args, **kwargs):
        module_list.append(self)
        return original_to(self, *args, **kwargs)

    try:
        torch.nn.Module.__init__ = patched_init
        torch.nn.Module.to = patched_to
        yield
    finally:
        torch.nn.Module.__init__ = original_init
        torch.nn.Module.to = original_to

    start = time.perf_counter()
    module_list = set(module_list)

    for module in module_list:
        module.cpu()

    model_management.soft_empty_cache()
    end = time.perf_counter()

    print(f'Automatic Memory Management: {len(module_list)} Modules in {(end - start):.2f} seconds.')
    return
