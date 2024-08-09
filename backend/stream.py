import torch
from backend.args import args


def stream_context():
    if torch.cuda.is_available():
        return torch.cuda.stream

    if torch.xpu.is_available():
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
        if torch.xpu.is_available():
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
        if torch.xpu.is_available():
            device = torch.device("xpu")
            stream = torch.xpu.Stream(device)
            with torch.xpu.stream(stream):
                torch.zeros((1, 1)).to(device, torch.float32)
            stream.synchronize()
            return stream
    except:
        return None


def should_use_stream():
    return stream_activated and current_stream is not None and mover_stream is not None


current_stream = get_current_stream()
mover_stream = get_new_stream()
stream_activated = args.cuda_stream
