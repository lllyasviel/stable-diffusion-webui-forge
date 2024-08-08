import os
import torch
import logging
import importlib

import huggingface_guess

from diffusers import DiffusionPipeline
from transformers import modeling_utils

from backend import memory_management
from backend.utils import read_arbitrary_config
from backend.state_dict import try_filter_state_dict, load_state_dict
from backend.operations import using_forge_operations
from backend.nn.vae import IntegratedAutoencoderKL
from backend.nn.clip import IntegratedCLIP
from backend.nn.unet import IntegratedUNet2DConditionModel

from backend.diffusion_engine.sd15 import StableDiffusion
from backend.diffusion_engine.sd20 import StableDiffusion2
from backend.diffusion_engine.sdxl import StableDiffusionXL
from backend.diffusion_engine.flux import Flux


possible_models = [StableDiffusion, StableDiffusion2, StableDiffusionXL, Flux]


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

            with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype()):
                model = IntegratedAutoencoderKL.from_config(config)

            load_state_dict(model, state_dict, ignore_start='loss.')
            return model
        if component_name.startswith('text_encoder') and cls_name in ['CLIPTextModel', 'CLIPTextModelWithProjection']:
            from transformers import CLIPTextConfig, CLIPTextModel
            config = CLIPTextConfig.from_pretrained(config_path)

            to_args = dict(device=memory_management.text_encoder_device(), dtype=memory_management.text_encoder_dtype())

            with modeling_utils.no_init_weights():
                with using_forge_operations(**to_args, manual_cast_enabled=True):
                    model = IntegratedCLIP(CLIPTextModel, config, add_text_projection=True).to(**to_args)

            load_state_dict(model, state_dict, ignore_errors=[
                'transformer.text_projection.weight',
                'transformer.text_model.embeddings.position_ids',
                'logit_scale'
            ], log_name=cls_name)

            return model
        if cls_name == 'T5EncoderModel':
            from backend.nn.t5 import IntegratedT5
            config = read_arbitrary_config(config_path)

            dtype = memory_management.text_encoder_dtype()
            sd_dtype = state_dict['transformer.encoder.block.0.layer.0.SelfAttention.k.weight'].dtype

            if sd_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                dtype = sd_dtype

            with modeling_utils.no_init_weights():
                with using_forge_operations(device=memory_management.cpu, dtype=dtype, manual_cast_enabled=True):
                    model = IntegratedT5(config)

            load_state_dict(model, state_dict, log_name=cls_name, ignore_errors=['transformer.encoder.embed_tokens.weight'])

            return model
        if cls_name == 'UNet2DConditionModel':
            unet_config = guess.unet_config.copy()
            state_dict_size = memory_management.state_dict_size(state_dict)
            ini_dtype = memory_management.unet_dtype(model_params=state_dict_size)
            ini_device = memory_management.unet_inital_load_device(parameters=state_dict_size, dtype=ini_dtype)
            to_args = dict(device=ini_device, dtype=ini_dtype)

            with using_forge_operations(**to_args):
                model = IntegratedUNet2DConditionModel.from_config(unet_config).to(**to_args)
                model._internal_dict = unet_config

            load_state_dict(model, state_dict)
            return model
        if cls_name == 'FluxTransformer2DModel':
            from backend.nn.flux import IntegratedFluxTransformer2DModel
            unet_config = guess.unet_config.copy()
            state_dict_size = memory_management.state_dict_size(state_dict)
            ini_dtype = memory_management.unet_dtype(model_params=state_dict_size)
            ini_device = memory_management.unet_inital_load_device(parameters=state_dict_size, dtype=ini_dtype)
            to_args = dict(device=ini_device, dtype=ini_dtype)

            with using_forge_operations(**to_args):
                model = IntegratedFluxTransformer2DModel(**unet_config).to(**to_args)
                model.config = unet_config

            load_state_dict(model, state_dict)
            return model

    print(f'Skipped: {component_name} = {lib_name}.{cls_name}')
    return None


def split_state_dict(sd, sd_vae=None):
    guess = huggingface_guess.guess(sd)
    guess.clip_target = guess.clip_target(sd)

    if sd_vae is not None:
        print(f'Using external VAE state dict: {len(sd_vae)}')

    state_dict = {
        guess.unet_target: try_filter_state_dict(sd, guess.unet_key_prefix),
        guess.vae_target: try_filter_state_dict(sd, guess.vae_key_prefix) if sd_vae is None else sd_vae
    }

    sd = guess.process_clip_state_dict(sd)

    for k, v in guess.clip_target.items():
        state_dict[v] = try_filter_state_dict(sd, [k + '.'])

    state_dict['ignore'] = sd

    print_dict = {k: len(v) for k, v in state_dict.items()}
    print(f'StateDict Keys: {print_dict}')

    del state_dict['ignore']

    return state_dict, guess


@torch.no_grad()
def forge_loader(sd, sd_vae=None):
    state_dicts, estimated_config = split_state_dict(sd, sd_vae=sd_vae)
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
