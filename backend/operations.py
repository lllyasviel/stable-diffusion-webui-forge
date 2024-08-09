import time
import torch
import contextlib

from backend import stream, memory_management


stash = {}


def weights_manual_cast(layer, x, skip_dtype=False):
    weight, bias, signal = None, None, None
    non_blocking = True

    if getattr(x.device, 'type', None) == 'mps':
        non_blocking = False

    target_dtype = x.dtype
    target_device = x.device

    if skip_dtype:
        target_dtype = None

    if stream.should_use_stream():
        with stream.stream_context()(stream.mover_stream):
            if layer.weight is not None:
                weight = layer.weight.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
            if layer.bias is not None:
                bias = layer.bias.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
            signal = stream.mover_stream.record_event()
    else:
        if layer.weight is not None:
            weight = layer.weight.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
        if layer.bias is not None:
            bias = layer.bias.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    return weight, bias, signal


@contextlib.contextmanager
def main_stream_worker(weight, bias, signal):
    if signal is None or not stream.should_use_stream():
        yield
        return

    with stream.stream_context()(stream.current_stream):
        stream.current_stream.wait_event(signal)
        yield
        finished_signal = stream.current_stream.record_event()
        stash[id(finished_signal)] = (weight, bias, finished_signal)

    garbage = []
    for k, (w, b, s) in stash.items():
        if s.query():
            garbage.append(k)

    for k in garbage:
        del stash[k]
    return


def cleanup_cache():
    if not stream.should_use_stream():
        return

    stream.current_stream.synchronize()
    stream.mover_stream.synchronize()
    stash.clear()
    return


current_device = None
current_dtype = None
current_manual_cast_enabled = False


class ForgeOperations:
    class Linear(torch.nn.Linear):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.linear(x, weight, bias)
            else:
                return super().forward(x)

    class Conv2d(torch.nn.Conv2d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias)
            else:
                return super().forward(x)

    class Conv3d(torch.nn.Conv3d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias)
            else:
                return super().forward(x)

    class Conv1d(torch.nn.Conv1d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias)
            else:
                return super().forward(x)

    class ConvTranspose2d(torch.nn.ConvTranspose2d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            if self.parameters_manual_cast:
                num_spatial_dims = 2
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)

                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.conv_transpose2d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
            else:
                return super().forward(x, output_size)

    class ConvTranspose1d(torch.nn.ConvTranspose1d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            if self.parameters_manual_cast:
                num_spatial_dims = 1
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)

                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.conv_transpose1d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
            else:
                return super().forward(x, output_size)

    class ConvTranspose3d(torch.nn.ConvTranspose3d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            if self.parameters_manual_cast:
                num_spatial_dims = 3
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)

                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.conv_transpose3d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
            else:
                return super().forward(x, output_size)

    class GroupNorm(torch.nn.GroupNorm):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.group_norm(x, self.num_groups, weight, bias, self.eps)
            else:
                return super().forward(x)

    class LayerNorm(torch.nn.LayerNorm):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
            else:
                return super().forward(x)

    class Embedding(torch.nn.Embedding):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled
            self.bias = None

        def reset_parameters(self):
            self.bias = None
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x, skip_dtype=True)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.embedding(x, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
            else:
                return super().forward(x)


@contextlib.contextmanager
def using_forge_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False):
    global current_device, current_dtype, current_manual_cast_enabled

    current_device, current_dtype, current_manual_cast_enabled = device, dtype, manual_cast_enabled

    if operations is None:
        operations = ForgeOperations

    op_names = ['Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'GroupNorm', 'LayerNorm', 'Embedding']
    backups = {op_name: getattr(torch.nn, op_name) for op_name in op_names}

    try:
        for op_name in op_names:
            setattr(torch.nn, op_name, getattr(operations, op_name))

        yield

    finally:
        for op_name in op_names:
            setattr(torch.nn, op_name, backups[op_name])
    return


def shift_manual_cast(model, enabled):
    for m in model.modules():
        if hasattr(m, 'parameters_manual_cast'):
            m.parameters_manual_cast = enabled
    return


@contextlib.contextmanager
def automatic_memory_management():
    memory_management.free_memory(
        memory_required=3 * 1024 * 1024 * 1024,
        device=memory_management.get_torch_device()
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

    memory_management.soft_empty_cache()
    end = time.perf_counter()

    print(f'Automatic Memory Management: {len(module_list)} Modules in {(end - start):.2f} seconds.')
    return
