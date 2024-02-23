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
            return torch.cuda.current_stream(torch.device(torch.cuda.current_device()))
        if model_management.is_intel_xpu():
            return torch.xpu.current_stream(torch.device("xpu"))
    except:
        pass
    print('Stream is not used.')
    return None


def get_new_stream():
    try:
        if torch.cuda.is_available():
            return torch.cuda.Stream(torch.device(torch.cuda.current_device()))
        if model_management.is_intel_xpu():
            return torch.xpu.Stream(torch.device("xpu"))
    except:
        pass
    print('Stream is not used.')
    return None


if shared.opts.use_non_streamlined_lowvram:
    current_stream = None
    mover_stream = None
    using_stream = False
else:
    current_stream = get_current_stream()
    mover_stream = get_new_stream()
    using_stream = current_stream is not None and mover_stream is not None
