import torch
import contextlib

from backend import stream


stash = {}


def weights_manual_cast(layer, x):
    weight, bias, signal = None, None, None
    non_blocking = True

    if getattr(x.device, 'type', None) == 'mps':
        non_blocking = False

    if stream.using_stream:
        with stream.stream_context()(stream.mover_stream):
            if layer.bias is not None:
                bias = layer.bias.to(device=x.device, dtype=x.dtype, non_blocking=non_blocking)
            weight = layer.weight.to(device=x.device, dtype=x.dtype, non_blocking=non_blocking)
            signal = stream.mover_stream.record_event()
    else:
        if layer.bias is not None:
            bias = layer.bias.to(device=x.device, dtype=x.dtype, non_blocking=non_blocking)
        weight = layer.weight.to(device=x.device, dtype=x.dtype, non_blocking=non_blocking)

    return weight, bias, signal


@contextlib.contextmanager
def main_stream_worker(weight, bias, signal):
    if not stream.using_stream or signal is None:
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
    if not stream.using_stream:
        return

    stream.current_stream.synchronize()
    stream.mover_stream.synchronize()
    stash.clear()
    return


class ForgeOperations:
    class Linear(torch.nn.Linear):
        parameters_manual_cast = False

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
        parameters_manual_cast = False

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
        parameters_manual_cast = False

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias)
            else:
                return super().forward(x)

    class GroupNorm(torch.nn.GroupNorm):
        parameters_manual_cast = False

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
        parameters_manual_cast = False

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
            else:
                return super().forward(x)


class ForgeOperationsWithManualCast(ForgeOperations):
    class Linear(ForgeOperations.Linear):
        parameters_manual_cast = True

    class Conv2d(ForgeOperations.Conv2d):
        parameters_manual_cast = True

    class Conv3d(ForgeOperations.Conv3d):
        parameters_manual_cast = True

    class GroupNorm(ForgeOperations.GroupNorm):
        parameters_manual_cast = True

    class LayerNorm(ForgeOperations.LayerNorm):
        parameters_manual_cast = True


@contextlib.contextmanager
def using_forge_operations(parameters_manual_cast=True, operations=None):

    if operations is None:
        operations = ForgeOperations

        if parameters_manual_cast:
            operations = ForgeOperationsWithManualCast

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
