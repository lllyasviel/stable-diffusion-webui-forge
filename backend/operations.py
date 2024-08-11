# Copyright Forge 2024

import time
import torch
import contextlib

from backend import stream, memory_management


stash = {}


def weights_manual_cast(layer, x, skip_weight_dtype=False, skip_bias_dtype=False):
    weight, bias, signal = None, None, None
    non_blocking = True

    if getattr(x.device, 'type', None) == 'mps':
        non_blocking = False

    target_dtype = x.dtype
    target_device = x.device

    if skip_weight_dtype:
        weight_args = dict(device=target_device, non_blocking=non_blocking)
    else:
        weight_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    if skip_bias_dtype:
        bias_args = dict(device=target_device, non_blocking=non_blocking)
    else:
        bias_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    if stream.should_use_stream():
        with stream.stream_context()(stream.mover_stream):
            if layer.weight is not None:
                weight = layer.weight.to(**weight_args)
            if layer.bias is not None:
                bias = layer.bias.to(**bias_args)
            signal = stream.mover_stream.record_event()
    else:
        if layer.weight is not None:
            weight = layer.weight.to(**weight_args)
        if layer.bias is not None:
            bias = layer.bias.to(**bias_args)

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
current_bnb_dtype = None


class ForgeOperations:
    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.empty(1, device=current_device, dtype=current_dtype))
            self.weight = None
            self.bias = None
            self.parameters_manual_cast = current_manual_cast_enabled

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if hasattr(self, 'dummy'):
                if prefix + 'weight' in state_dict:
                    self.weight = torch.nn.Parameter(state_dict[prefix + 'weight'].to(self.dummy))
                if prefix + 'bias' in state_dict:
                    self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))
                del self.dummy
            else:
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.linear(x, weight, bias)
            else:
                return torch.nn.functional.linear(x, self.weight, self.bias)

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
                weight, bias, signal = weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.embedding(x, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
            else:
                return super().forward(x)


try:
    from backend.operations_bnb import ForgeLoader4Bit, ForgeParams4bit, functional_linear_4bits

    class ForgeOperationsBNB4bits(ForgeOperations):
        class Linear(ForgeLoader4Bit):
            def __init__(self, *args, **kwargs):
                super().__init__(device=current_device, dtype=current_dtype, quant_type=current_bnb_dtype)
                self.parameters_manual_cast = current_manual_cast_enabled

            def forward(self, x):
                self.weight.quant_state = self.quant_state

                if self.bias is not None and self.bias.dtype != x.dtype:
                    # Maybe this can also be set to all non-bnb ops since the cost is very low.
                    # And it only invokes one time, and most linear does not have bias
                    self.bias.data = self.bias.data.to(x.dtype)

                if not self.parameters_manual_cast:
                    return functional_linear_4bits(x, self.weight, self.bias)
                elif not self.weight.bnb_quantized:
                    assert x.device.type == 'cuda', 'BNB Must Use CUDA as Computation Device!'
                    layer_original_device = self.weight.device
                    self.weight = self.weight._quantize(x.device)
                    bias = self.bias.to(x.device) if self.bias is not None else None
                    out = functional_linear_4bits(x, self.weight, bias)
                    self.weight = self.weight.to(layer_original_device)
                    return out
                else:
                    weight, bias, signal = weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)
                    with main_stream_worker(weight, bias, signal):
                        return functional_linear_4bits(x, weight, bias)

    bnb_avaliable = True
except:
    bnb_avaliable = False


@contextlib.contextmanager
def using_forge_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False, bnb_dtype=None):
    global current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype

    current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype = device, dtype, manual_cast_enabled, bnb_dtype

    if operations is None:
        if bnb_avaliable and bnb_dtype in ['nf4', 'fp4']:
            operations = ForgeOperationsBNB4bits
        else:
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
