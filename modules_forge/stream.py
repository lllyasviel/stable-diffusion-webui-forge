# https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14855

import torch

from modules import shared
from ldm_patched.modules import model_management


def stream_context():
    if torch.cuda.is_available():
        return torch.cuda.stream

    if model_management.is_intel_xpu():
        return torch.xpu.stream

    return None


def get_current_stream():
    try:
        if torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
            stream = torch.cuda.current_stream(device)
            with torch.cuda.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
        if model_management.is_intel_xpu():
            device = torch.device("xpu")
            stream = torch.xpu.current_stream(device)
            with torch.xpu.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
    except:
        return None


def get_new_stream():
    try:
        if torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
            stream = torch.cuda.Stream(device)
            with torch.cuda.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
        if model_management.is_intel_xpu():
            device = torch.device("xpu")
            stream = torch.xpu.Stream(device)
            with torch.xpu.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
    except:
        return None


if shared.opts.use_non_streamlined_lowvram:
    current_stream = None
    mover_stream = None
    using_stream = False
else:
    current_stream = get_current_stream()
    mover_stream = get_new_stream()
    using_stream = current_stream is not None and mover_stream is not None

if not using_stream:
    print('Stream is not used.')
