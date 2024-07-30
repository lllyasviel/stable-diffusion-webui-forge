from diffusers import AutoencoderKL
from backend.state_dict import split_state_dict_with_prefix, compile_state_dict
from backend.operations import using_forge_operations
from backend.attention import AttentionProcessorForge
from diffusers.loaders.single_file_model import convert_ldm_vae_checkpoint


class BaseAutoencoderKL(AutoencoderKL):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.state_dict_mapping = {}

    def encode(self, x, regulation=None, mode=False):
        latent_dist = super().encode(x).latent_dist
        if mode:
            return latent_dist.mode()
        elif regulation is not None:
            return regulation(latent_dist)
        else:
            return latent_dist.sample()

    def decode(self, x):
        return super().decode(x).sample


def load_vae(state_dict, config):
    with using_forge_operations():
        model = BaseAutoencoderKL(**config)

    vae_state_dict = split_state_dict_with_prefix(state_dict, "first_stage_model.")
    vae_state_dict = convert_ldm_vae_checkpoint(vae_state_dict, config)
    vae_state_dict, mapping = compile_state_dict(vae_state_dict)
    model.load_state_dict(vae_state_dict, strict=True)
    model.set_attn_processor(AttentionProcessorForge())
    model.state_dict_mapping = mapping

    return model
