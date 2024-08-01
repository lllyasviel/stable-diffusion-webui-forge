import os
import importlib

from diffusers.loaders.single_file_utils import fetch_diffusers_config
from diffusers import DiffusionPipeline
from backend.vae import load_vae


dir_path = os.path.dirname(__file__)


def load_component(component_name, lib_name, cls_name, repo_path, sd):
    config_path = os.path.join(repo_path, component_name)
    if component_name in ['scheduler', 'tokenizer']:
        cls = getattr(importlib.import_module(lib_name), cls_name)
        return cls.from_pretrained(os.path.join(repo_path, component_name))
    if cls_name in ['AutoencoderKL']:
        return load_vae(sd, config_path)
    return None


def guess_repo_name_from_state_dict(sd):
    result = fetch_diffusers_config(sd)['pretrained_model_name_or_path']
    return result


def load_huggingface_components(sd):
    repo_name = guess_repo_name_from_state_dict(sd)
    local_path = os.path.join(dir_path, 'huggingface', repo_name)
    config = DiffusionPipeline.load_config(local_path)
    result = {"repo_path": local_path}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component = load_component(component_name, lib_name, cls_name, local_path, sd)
            if component is not None:
                result[component_name] = component
    return result
