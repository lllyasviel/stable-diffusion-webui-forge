import os
import importlib
import diffusers
import transformers

from diffusers.loaders.single_file_utils import fetch_diffusers_config
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL
from backend.vae import load_vae


dir_path = os.path.dirname(__file__)


def load_component(component_name, lib_name, cls_name, repo_path, sd):
    if component_name in ['scheduler', 'tokenizer']:
        cls = getattr(importlib.import_module(lib_name), cls_name)
        return cls.from_pretrained(os.path.join(repo_path, component_name))
    if cls_name in ['AutoencoderKL']:
        config = AutoencoderKL.load_config(os.path.join(repo_path, component_name))
        return load_vae(sd, config)

    return None


def load_huggingface_components(sd):
    pretrained_model_name_or_path = fetch_diffusers_config(sd)['pretrained_model_name_or_path']
    local_path = os.path.join(dir_path, 'huggingface', pretrained_model_name_or_path)
    config = DiffusionPipeline.load_config(local_path)
    result = {"repo_path": local_path}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component = load_component(component_name, lib_name, cls_name, local_path, sd)
            if component is not None:
                result[component_name] = component
    return result
