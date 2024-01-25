import torch
import contextlib

from ldm_patched.modules import model_management
from ldm_patched.modules import model_detection

from ldm_patched.modules.sd import VAE, CLIP, load_model_weights
import ldm_patched.modules.model_patcher
import ldm_patched.modules.utils
import ldm_patched.modules.clip_vision

from omegaconf import OmegaConf
from modules.sd_models_config import find_checkpoint_config
from modules.shared import cmd_opts
from ldm.util import instantiate_from_config

import open_clip
from transformers import CLIPTextModel, CLIPTokenizer


class FakeObject:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.visual = None
        return

    def eval(self, *args, **kwargs):
        return self

    def parameters(self, *args, **kwargs):
        return []


class ForgeSD:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet = unet
        self.clip = clip
        self.vae = vae
        self.clipvision = clipvision


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


def load_checkpoint_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True):
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = ldm_patched.modules.utils.calculate_parameters(sd, "model.diffusion_model.")
    unet_dtype = model_management.unet_dtype(model_params=parameters)
    load_device = model_management.get_torch_device()
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)

    class WeightsLoader(torch.nn.Module):
        pass

    model_config = model_detection.model_config_from_unet(sd, "model.diffusion_model.", unet_dtype)
    model_config.set_manual_cast(manual_cast_dtype)

    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type")

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = ldm_patched.modules.clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        offload_device = model_management.unet_offload_device()
        model = model_config.get_model(sd, "model.diffusion_model.", device=inital_load_device)
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae_sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {"first_stage_model.": ""}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)

    if output_clip:
        w = WeightsLoader()
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip = CLIP(clip_target, embedding_directory=embedding_directory)
            w.cond_stage_model = clip.cond_stage_model
            sd = model_config.process_clip_state_dict(sd)
            load_model_weights(w, sd)

    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    if output_model:
        model_patcher = ldm_patched.modules.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device(), current_device=inital_load_device)
        if inital_load_device != torch.device("cpu"):
            print("loaded straight to GPU")
            model_management.load_model_gpu(model_patcher)

    return ForgeSD(model_patcher, clip, vae, clipvision)


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

    forge_object = load_checkpoint_guess_config(
        state_dict,
        output_vae=True,
        output_clip=True,
        output_clipvision=True,
        embedding_directory=cmd_opts.embeddings_dir,
        output_model=True
    )
    sd_model.forge_objects = forge_object
    timer.record("forge load real models")

    sd_model.first_stage_model = forge_object.vae.first_stage_model
    sd_model.model.diffusion_model = forge_object.unet.model.diffusion_model

    conditioner = getattr(sd_model, 'conditioner', None)
    if conditioner:
        text_cond_models = []

        for i in range(len(conditioner.embedders)):
            embedder = conditioner.embedders[i]
            typename = type(embedder).__name__
            if typename == 'FrozenOpenCLIPEmbedder':
                embedder.tokenizer = forge_object.clip.tokenizer.clip_g.tokenizer
                embedder.transformer = forge_object.clip.cond_stage_model.clip_g.transformer
                text_cond_models.append(embedder)
            if typename == 'FrozenCLIPEmbedder':
                embedder.tokenizer = forge_object.clip.tokenizer.clip_l.tokenizer
                embedder.transformer = forge_object.clip.cond_stage_model.clip_l.transformer
                text_cond_models.append(embedder)
            if typename == 'FrozenOpenCLIPEmbedder2':
                embedder.tokenizer = forge_object.clip.tokenizer.clip_g.tokenizer
                embedder.transformer = forge_object.clip.cond_stage_model.clip_g.transformer
                text_cond_models.append(embedder)

        if len(text_cond_models) == 1:
            sd_model.cond_stage_model = text_cond_models[0]
        else:
            sd_model.cond_stage_model = conditioner
    elif type(sd_model.cond_stage_model).__name__ == 'FrozenCLIPEmbedder':
        sd_model.cond_stage_model.tokenizer = forge_object.clip.tokenizer.clip_l.tokenizer
        sd_model.cond_stage_model.transformer = forge_object.clip.cond_stage_model.clip_l.transformer
        pass
    elif type(sd_model.cond_stage_model).__name__ == 'FrozenOpenCLIPEmbedder':
        sd_model.cond_stage_model.tokenizer = forge_object.clip.tokenizer.clip_g.tokenizer
        sd_model.cond_stage_model.transformer = forge_object.clip.cond_stage_model.clip_g.transformer
        pass

    timer.record("forge set components")

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
