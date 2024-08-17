# Copyright Forge 2024

import torch
import bitsandbytes as bnb

from backend import utils
from bitsandbytes.nn.modules import Params4bit, QuantState
from bitsandbytes.functional import dequantize_4bit


def functional_linear_4bits(x, weight, bias):
    out = bnb.matmul_4bit(x, weight.t(), bias=bias, quant_state=weight.quant_state)
    out = out.to(x)
    return out


def functional_dequantize_4bit(weight):
    return dequantize_4bit(weight, quant_state=weight.quant_state, blocksize=weight.blocksize, quant_type=weight.quant_type)


def copy_quant_state(state: QuantState, device: torch.device = None) -> QuantState:
    if state is None:
        return None

    device = device or state.absmax.device

    state2 = (
        QuantState(
            absmax=state.state2.absmax.to(device),
            shape=state.state2.shape,
            code=state.state2.code.to(device),
            blocksize=state.state2.blocksize,
            quant_type=state.state2.quant_type,
            dtype=state.state2.dtype,
        )
        if state.nested
        else None
    )

    return QuantState(
        absmax=state.absmax.to(device),
        shape=state.shape,
        code=state.code.to(device),
        blocksize=state.blocksize,
        quant_type=state.quant_type,
        dtype=state.dtype,
        offset=state.offset.to(device) if state.nested else None,
        state2=state2,
    )


class ForgeParams4bit(Params4bit):
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type == "cuda" and not self.bnb_quantized:
            return self._quantize(device)
        else:
            return ForgeParams4bit(
                torch.nn.Parameter.to(self, device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
                quant_state=copy_quant_state(self.quant_state, device),
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                bnb_quantized=self.bnb_quantized,
            )

    def pin_memory(self, device=None):
        return ForgeParams4bit(
            torch.Tensor.pin_memory(self, device=device),
            requires_grad=self.requires_grad,
            quant_state=self.quant_state,
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
            bnb_quantized=self.bnb_quantized,
        )


class ForgeLoader4Bit(torch.nn.Module):
    def __init__(self, *, device, dtype, quant_type, **kwargs):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        self.weight = None
        self.bias = None
        self.quant_type = quant_type

    def _apply(self, fn, recurse=True):
        if self.weight is not None:
            self.weight = utils.tensor2parameter(fn(self.weight))
        if self.bias is not None:
            self.bias = utils.tensor2parameter(fn(self.bias))
        return self

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        quant_state = getattr(self.weight, "quant_state", None)
        if quant_state is not None:
            for k, v in quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()
        return

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        quant_state_keys = {k[len(prefix + "weight."):] for k in state_dict.keys() if k.startswith(prefix + "weight.")}

        if any('bitsandbytes' in k for k in quant_state_keys):
            quant_state_dict = {k: state_dict[prefix + "weight." + k] for k in quant_state_keys}

            self.weight = ForgeParams4bit.from_prequantized(
                data=state_dict[prefix + 'weight'],
                quantized_stats=quant_state_dict,
                requires_grad=False,
                device=self.dummy.device,
            )

            if prefix + 'bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))

            del self.dummy
        elif hasattr(self, 'dummy'):
            if prefix + 'weight' in state_dict:
                self.weight = ForgeParams4bit(
                    state_dict[prefix + 'weight'].to(self.dummy),
                    requires_grad=False,
                    compress_statistics=False,
                    blocksize=64,
                    quant_type=self.quant_type,
                    quant_storage=torch.uint8,
                )

            if prefix + 'bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))

            del self.dummy
        else:
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def reload_weight(self, weight):
        self.weight = ForgeParams4bit(
            weight,
            requires_grad=False,
            compress_statistics=self.weight.compress_statistics,
            blocksize=self.weight.blocksize,
            quant_type=self.weight.quant_type,
            quant_storage=self.weight.quant_storage,
            bnb_quantized=False
        )
        return self
