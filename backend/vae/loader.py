from diffusers import AutoencoderKL
from backend.vae.configs.guess import guess_vae_config
from backend.state_dict import StateDictItem, compile_state_dict
from backend.operations import using_forge_operations
from backend.attention import AttentionProcessorForge
from diffusers.loaders.single_file_model import convert_ldm_vae_checkpoint


def convert_vae_state_dict(sd):
    vae_sd = {}
    prefix = "first_stage_model."

    for k, v in list(sd.items()):
        if k.startswith(prefix):
            vae_sd[k] = StateDictItem(k[len(prefix):], v)
            del sd[k]

    return vae_sd


def load_vae_from_state_dict(state_dict):
    config = guess_vae_config(state_dict)

    with using_forge_operations():
        model = AutoencoderKL(**config)

    vae_state_dict = convert_vae_state_dict(state_dict)
    vae_state_dict = convert_ldm_vae_checkpoint(vae_state_dict, config)
    vae_state_dict, mapping = compile_state_dict(vae_state_dict)
    model.load_state_dict(vae_state_dict, strict=True)
    model.set_attn_processor(AttentionProcessorForge())

    return model, mapping
