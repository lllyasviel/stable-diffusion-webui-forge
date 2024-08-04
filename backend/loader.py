import os
import logging
import importlib
import huggingface_guess

from diffusers import DiffusionPipeline
from transformers import modeling_utils
from backend.state_dict import try_filter_state_dict, transformers_convert, load_state_dict, state_dict_key_replace
from backend.operations import using_forge_operations
from backend.nn.vae import IntegratedAutoencoderKL
from backend.nn.clip import IntegratedCLIP, CLIPTextConfig
from backend.nn.unet import IntegratedUNet2DConditionModel


logging.getLogger("diffusers").setLevel(logging.ERROR)
dir_path = os.path.dirname(__file__)


def load_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
    config_path = os.path.join(repo_path, component_name)

    if component_name in ['feature_extractor', 'safety_checker']:
        return None

    if lib_name in ['transformers', 'diffusers']:
        if component_name in ['scheduler'] or component_name.startswith('tokenizer'):
            cls = getattr(importlib.import_module(lib_name), cls_name)
            return cls.from_pretrained(os.path.join(repo_path, component_name))
        if cls_name in ['AutoencoderKL']:
            sd = try_filter_state_dict(state_dict, ['first_stage_model.', 'vae.'])
            config = IntegratedAutoencoderKL.load_config(config_path)

            with using_forge_operations():
                model = IntegratedAutoencoderKL.from_config(config)

            load_state_dict(model, sd)
            return model
        if component_name.startswith('text_encoder') and cls_name in ['CLIPTextModel', 'CLIPTextModelWithProjection']:
            if component_name == 'text_encoder':
                sd = try_filter_state_dict(state_dict, ['cond_stage_model.', 'conditioner.embedders.0.'])
            elif component_name == 'text_encoder_2':
                sd = try_filter_state_dict(state_dict, ['conditioner.embedders.1.'])
            else:
                raise ValueError(f"Wrong component_name: {component_name}")

            if 'model.text_projection' in sd:
                sd = transformers_convert(sd, "model.", "transformer.text_model.", 32)
                sd = state_dict_key_replace(sd, {"model.text_projection": "text_projection",
                                                 "model.text_projection.weight": "text_projection",
                                                 "model.logit_scale": "logit_scale"})

            config = CLIPTextConfig.from_pretrained(config_path)

            with modeling_utils.no_init_weights():
                with using_forge_operations():
                    model = IntegratedCLIP(config)

            load_state_dict(model, sd, ignore_errors=['text_projection', 'logit_scale',
                                                      'transformer.text_model.embeddings.position_ids'])
            return model
        if cls_name == 'UNet2DConditionModel':
            sd = try_filter_state_dict(state_dict, ['model.diffusion_model.'])

            with using_forge_operations():
                model = IntegratedUNet2DConditionModel.from_config(guess.unet_config)
                model._internal_dict = guess.unet_config

            load_state_dict(model, sd)
            return model

    print(f'Skipped: {component_name} = {lib_name}.{cls_name}')
    return None


def load_huggingface_components(sd):
    guess = huggingface_guess.guess(sd)
    repo_name = guess.huggingface_repo
    local_path = os.path.join(dir_path, 'huggingface', repo_name)
    config = DiffusionPipeline.load_config(local_path)
    result = {"repo_path": local_path}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component = load_component(guess, component_name, lib_name, cls_name, local_path, sd)
            if component is not None:
                result[component_name] = component
    return result
