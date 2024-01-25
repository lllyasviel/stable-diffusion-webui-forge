import torch
import contextlib

from ldm_patched.modules import model_management
from ldm_patched.modules import model_detection

from ldm_patched.modules.sd import VAE
import ldm_patched.modules.model_patcher
import ldm_patched.modules.utils

from omegaconf import OmegaConf
from modules.sd_models_config import find_checkpoint_config
from ldm.util import instantiate_from_config

import open_clip
from transformers import CLIPTextModel, CLIPTokenizer


class FakeObject(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.visual = None
        return


@contextlib.contextmanager
def no_clip():
    backup_openclip = open_clip.create_model_and_transforms
    backup_CLIPTextModel = CLIPTextModel.from_pretrained
    backup_CLIPTokenizer = CLIPTokenizer.from_pretrained

    try:
        open_clip.create_model_and_transforms = lambda *args, **kwargs: (FakeObject(), None, None)
        CLIPTextModel.from_pretrained = lambda *args, **kwargs: FakeObject()
        CLIPTokenizer.from_pretrained = lambda *args, **kwargs: FakeObject()
        yield

    finally:
        open_clip.create_model_and_transforms = backup_openclip
        CLIPTextModel.from_pretrained = backup_CLIPTextModel
        CLIPTokenizer.from_pretrained = backup_CLIPTokenizer
    return


def load_model_for_a1111(timer, checkpoint_info=None, state_dict=None):
    a1111_config = find_checkpoint_config(state_dict, checkpoint_info)
    a1111_config = OmegaConf.load(a1111_config)
    timer.record("forge solving config")

    if hasattr(a1111_config.model.params, 'network_config'):
        a1111_config.model.params.network_config.target = 'modules_forge.forge_loader.FakeObject'

    if hasattr(a1111_config.model.params, 'unet_config'):
        a1111_config.model.params.unet_config.target = 'modules_forge.forge_loader.FakeObject'

    if hasattr(a1111_config.model.params, 'first_stage_config'):
        a1111_config.model.params.first_stage_config.target = 'modules_forge.forge_loader.FakeObject'

    with no_clip():
        sd_model = instantiate_from_config(a1111_config.model)

    timer.record("forge instantiate config")

    return


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
