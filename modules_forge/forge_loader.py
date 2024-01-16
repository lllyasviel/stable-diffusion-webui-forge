import torch

from ldm_patched.modules import model_management
from ldm_patched.modules import model_detection

from ldm_patched.modules.sd import VAE
import ldm_patched.modules.model_patcher
import ldm_patched.modules.utils


def load_unet_and_vae(sd):
    parameters = ldm_patched.modules.utils.calculate_parameters(sd, "model.diffusion_model.")
    unet_dtype = model_management.unet_dtype(model_params=parameters)
    load_device = model_management.get_torch_device()
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)

    model_config = model_detection.model_config_from_unet(sd, "model.diffusion_model.", unet_dtype)
    model_config.set_manual_cast(manual_cast_dtype)

    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of")

    initial_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
    model = model_config.get_model(sd, "model.diffusion_model.", device=initial_load_device)
    model.load_model_weights(sd, "model.diffusion_model.")

    model_patcher = ldm_patched.modules.model_patcher.ModelPatcher(model,
                                                                   load_device=load_device,
                                                                   offload_device=model_management.unet_offload_device(),
                                                                   current_device=initial_load_device)

    vae_sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {"first_stage_model.": ""}, filter_keys=True)
    vae_sd = model_config.process_vae_state_dict(vae_sd)
    vae_patcher = VAE(sd=vae_sd)

    return model_patcher, vae_patcher


class FakeObject(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        return
