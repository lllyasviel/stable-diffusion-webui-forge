import os
import ldm_patched.modules.utils

from modules.paths_internal import models_path


controlnet_dir = os.path.join(models_path, 'ControlNet')
os.makedirs(controlnet_dir, exist_ok=True)

preprocessor_dir = os.path.join(models_path, 'ControlNetPreprocessor')
os.makedirs(preprocessor_dir, exist_ok=True)

diffusers_dir = os.path.join(models_path, 'diffusers')
os.makedirs(diffusers_dir, exist_ok=True)

supported_preprocessors = {}
supported_control_models = []


def add_supported_preprocessor(preprocessor):
    global supported_preprocessors
    p = preprocessor
    supported_preprocessors[p.name] = p
    return


def add_supported_control_model(control_model):
    global supported_control_models
    supported_control_models.append(control_model)
    return


def try_load_supported_control_model(ckpt_path):
    global supported_control_models
    state_dict = ldm_patched.modules.utils.load_torch_file(ckpt_path, safe_load=True)
    for supported_type in supported_control_models:
        state_dict_copy = dict(state_dict)
        model = supported_type.try_build_from_state_dict(state_dict_copy, ckpt_path)
        if model is not None:
            return model
    return None
