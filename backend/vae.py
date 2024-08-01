from backend.state_dict import filter_state_dict_with_prefix
from backend.operations import using_forge_operations
from backend.nn.autoencoder_kl import IntegratedAutoencoderKL


def load_vae(state_dict, config_path):
    config = IntegratedAutoencoderKL.load_config(config_path)

    with using_forge_operations():
        model = IntegratedAutoencoderKL.from_config(config)

    vae_state_dict = filter_state_dict_with_prefix(state_dict, "first_stage_model.")
    model.load_state_dict(vae_state_dict, strict=True)
    return model
