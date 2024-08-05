import os
import torch
import logging
import importlib
import huggingface_guess

from diffusers import DiffusionPipeline
from transformers import modeling_utils
from backend.state_dict import try_filter_state_dict, load_state_dict
from backend.operations import using_forge_operations
from backend.nn.vae import IntegratedAutoencoderKL
from backend.nn.clip import IntegratedCLIP, CLIPTextConfig
from backend.nn.unet import IntegratedUNet2DConditionModel

from backend.diffusion_engine.sd15 import StableDiffusion
from backend.diffusion_engine.sdxl import StableDiffusionXL


possible_models = [StableDiffusion, StableDiffusionXL]


logging.getLogger("diffusers").setLevel(logging.ERROR)
dir_path = os.path.dirname(__file__)


def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
    config_path = os.path.join(repo_path, component_name)

    if component_name in ['feature_extractor', 'safety_checker']:
        return None

    if lib_name in ['transformers', 'diffusers']:
        if component_name in ['scheduler'] or component_name.startswith('tokenizer'):
            cls = getattr(importlib.import_module(lib_name), cls_name)
            return cls.from_pretrained(os.path.join(repo_path, component_name))
        if cls_name in ['AutoencoderKL']:
            config = IntegratedAutoencoderKL.load_config(config_path)

            with using_forge_operations():
                model = IntegratedAutoencoderKL.from_config(config)

            load_state_dict(model, state_dict)
            return model
        if component_name.startswith('text_encoder') and cls_name in ['CLIPTextModel', 'CLIPTextModelWithProjection']:
            config = CLIPTextConfig.from_pretrained(config_path)

            with modeling_utils.no_init_weights():
                with using_forge_operations():
                    model = IntegratedCLIP(config)

            load_state_dict(model, state_dict, ignore_errors=[
                'transformer.text_projection.weight',
                'transformer.text_model.embeddings.position_ids',
                'logit_scale'
            ], log_name=cls_name)

            return model
        if cls_name == 'UNet2DConditionModel':
            with using_forge_operations():
                model = IntegratedUNet2DConditionModel.from_config(guess.unet_config)
                model._internal_dict = guess.unet_config

            load_state_dict(model, state_dict)
            return model

    print(f'Skipped: {component_name} = {lib_name}.{cls_name}')
    return None


def split_state_dict(sd):
    guess = huggingface_guess.guess(sd)

    state_dict = {
        'unet': try_filter_state_dict(sd, ['model.diffusion_model.']),
        'vae': try_filter_state_dict(sd, guess.vae_key_prefix)
    }

    sd = guess.process_clip_state_dict(sd)
    guess.clip_target = guess.clip_target(sd)

    for k, v in guess.clip_target.items():
        state_dict[v] = try_filter_state_dict(sd, [k + '.'])

    state_dict['ignore'] = sd

    print_dict = {k: len(v) for k, v in state_dict.items()}
    print(f'StateDict Keys: {print_dict}')

    del state_dict['ignore']

    return state_dict, guess


@torch.no_grad()
def forge_loader(sd):
    state_dicts, estimated_config = split_state_dict(sd)
    repo_name = estimated_config.huggingface_repo

    local_path = os.path.join(dir_path, 'huggingface', repo_name)
    config: dict = DiffusionPipeline.load_config(local_path)
    huggingface_components = {}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component_sd = state_dicts.get(component_name, None)
            component = load_huggingface_component(estimated_config, component_name, lib_name, cls_name, local_path, component_sd)
            if component_sd is not None:
                del state_dicts[component_name]
            if component is not None:
                huggingface_components[component_name] = component

    for M in possible_models:
        if any(isinstance(estimated_config, x) for x in M.matched_guesses):
            return M(estimated_config=estimated_config, huggingface_components=huggingface_components)

    print('Failed to recognize model type!')
    return None
