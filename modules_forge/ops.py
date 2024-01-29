import torch
import contextlib


@contextlib.contextmanager
def use_patched_ops(operations):
    op_names = ['Linear', 'Conv2d', 'Conv3d', 'GroupNorm', 'LayerNorm']
    backups = {op_name: getattr(torch.nn, op_name) for op_name in op_names}

    try:
        for op_name in op_names:
            setattr(torch.nn, op_name, getattr(operations, op_name))

        yield

    finally:
        for op_name in op_names:
            setattr(torch.nn, op_name, backups[op_name])
    return


@contextlib.contextmanager
def capture_model():
    module_list = []
    backup_init = torch.nn.Module.__init__

    def patched_init(self, *args, **kwargs):
        module_list.append(self)
        return backup_init(self, *args, **kwargs)

    try:
        torch.nn.Module.__init__ = patched_init
        yield
    finally:
        torch.nn.Module.__init__ = backup_init

    results = []
    for item in module_list:
        item_params = getattr(item, '_parameters', [])
        if len(item_params) > 0:
            results.append(item)

    if len(results) == 0:
        return None

    captured_model = torch.nn.ModuleList(results)

    return captured_model
