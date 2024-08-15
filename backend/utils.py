import gguf
import torch
import os
import json
import safetensors.torch
import backend.misc.checkpoint_pickle
from backend.operations_gguf import ParameterGGUF


def read_arbitrary_config(directory):
    config_path = os.path.join(directory, 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json file found in the directory: {directory}")

    with open(config_path, 'rt', encoding='utf-8') as file:
        config_data = json.load(file)

    return config_data


def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    elif ckpt.lower().endswith(".gguf"):
        reader = gguf.GGUFReader(ckpt)
        sd = {}
        for tensor in reader.tensors:
            sd[str(tensor.name)] = ParameterGGUF(tensor)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=backend.misc.checkpoint_pickle)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], torch.nn.Parameter(value, requires_grad=False))


def set_attr_raw(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)


def copy_to_param(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def fp16_fix(x):
    # An interesting trick to avoid fp16 overflow
    # Source: https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/1114
    # Related: https://github.com/comfyanonymous/ComfyUI/blob/f1d6cef71c70719cc3ed45a2455a4e5ac910cd5e/comfy/ldm/flux/layers.py#L180

    if x.dtype in [torch.float16]:
        return x.clip(-32768.0, 32768.0)
    return x
