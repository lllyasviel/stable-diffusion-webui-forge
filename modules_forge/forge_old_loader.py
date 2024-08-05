import torch
import contextlib

from backend import memory_management, utils
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.base import ModelPatcher
import backend.nn.unet

from omegaconf import OmegaConf
from modules.sd_models_config import find_checkpoint_config
from modules.shared import cmd_opts, opts
from modules import sd_hijack
from modules.sd_models_xl import extend_sdxl
from ldm.util import instantiate_from_config
from modules_forge import clip
from modules_forge.unet_patcher import UnetPatcher
from backend.loader import load_huggingface_components
from backend.modules.k_model import KModel
from backend.text_processing.classic_engine import ClassicTextProcessingEngine

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

    def shallow_copy(self):
        return ForgeSD(
            self.unet,
            self.clip,
            self.vae,
            self.clipvision
        )


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

    parameters = utils.calculate_parameters(sd, "model.diffusion_model.")
    unet_dtype = memory_management.unet_dtype(model_params=parameters)
    load_device = memory_management.get_torch_device()
    manual_cast_dtype = memory_management.unet_manual_cast(unet_dtype, load_device)
    manual_cast_dtype = unet_dtype if manual_cast_dtype is None else manual_cast_dtype

    initial_load_device = memory_management.unet_inital_load_device(parameters, unet_dtype)
    backend.nn.unet.unet_initial_device = initial_load_device
    backend.nn.unet.unet_initial_dtype = unet_dtype

    huggingface_components = load_huggingface_components(sd)

    if output_model:
        k_model = KModel(huggingface_components, storage_dtype=unet_dtype, computation_dtype=manual_cast_dtype)
        k_model.to(device=initial_load_device, dtype=unet_dtype)
        model_patcher = UnetPatcher(k_model, load_device=load_device,
                                    offload_device=memory_management.unet_offload_device(),
                                    current_device=initial_load_device)

    if output_vae:
        vae = huggingface_components['vae']
        vae = VAE(model=vae)

    if output_clip:
        clip = CLIP(huggingface_components)

    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    return ForgeSD(model_patcher, clip, vae, clipvision)


@torch.no_grad()
def load_model_for_a1111(timer, checkpoint_info=None, state_dict=None):
    a1111_config_filename = find_checkpoint_config(state_dict, checkpoint_info)
    a1111_config = OmegaConf.load(a1111_config_filename)
    timer.record("forge solving config")

    if hasattr(a1111_config.model.params, 'network_config'):
        a1111_config.model.params.network_config.target = 'modules_forge.loader.FakeObject'

    if hasattr(a1111_config.model.params, 'unet_config'):
        a1111_config.model.params.unet_config.target = 'modules_forge.loader.FakeObject'

    if hasattr(a1111_config.model.params, 'first_stage_config'):
        a1111_config.model.params.first_stage_config.target = 'modules_forge.loader.FakeObject'

    with no_clip():
        sd_model = instantiate_from_config(a1111_config.model)

    timer.record("forge instantiate config")

    forge_objects = load_checkpoint_guess_config(
        state_dict,
        output_vae=True,
        output_clip=True,
        output_clipvision=True,
        embedding_directory=cmd_opts.embeddings_dir,
        output_model=True
    )
    sd_model.forge_objects = forge_objects
    sd_model.forge_objects_original = forge_objects.shallow_copy()
    sd_model.forge_objects_after_applying_lora = forge_objects.shallow_copy()
    timer.record("forge load real models")

    sd_model.first_stage_model = forge_objects.vae.first_stage_model
    sd_model.model.diffusion_model = forge_objects.unet.model

    def set_clip_skip_callback(m, ts):
        m.clip_skip = opts.CLIP_stop_at_last_layers
        return

    def set_clip_skip_callback_and_move_model(m, ts):
        memory_management.load_model_gpu(sd_model.forge_objects.clip.patcher)
        m.clip_skip = opts.CLIP_stop_at_last_layers
        return

    conditioner = getattr(sd_model, 'conditioner', None)
    if conditioner:
        text_cond_models = []

        for i in range(len(conditioner.embedders)):
            embedder = conditioner.embedders[i]
            typename = type(embedder).__name__
            if typename == 'FrozenCLIPEmbedder':  # SDXL Clip L
                engine = ClassicTextProcessingEngine(
                    text_encoder=forge_objects.clip.cond_stage_model.clip_l,
                    tokenizer=forge_objects.clip.tokenizer.clip_l,
                    embedding_dir=cmd_opts.embeddings_dir,
                    embedding_key='clip_l',
                    embedding_expected_shape=2048,
                    emphasis_name=opts.emphasis,
                    text_projection=False,
                    minimal_clip_skip=2,
                    clip_skip=2,
                    return_pooled=False,
                    final_layer_norm=False,
                    callback_before_encode=set_clip_skip_callback
                )
                engine.is_trainable = False  # for sgm codebase
                engine.legacy_ucg_val = None  # for sgm codebase
                engine.input_key = 'txt'  # for sgm codebase
                conditioner.embedders[i] = engine
                text_cond_models.append(embedder)
            elif typename == 'FrozenOpenCLIPEmbedder2':  # SDXL Clip G
                engine = ClassicTextProcessingEngine(
                    text_encoder=forge_objects.clip.cond_stage_model.clip_g,
                    tokenizer=forge_objects.clip.tokenizer.clip_g,
                    embedding_dir=cmd_opts.embeddings_dir,
                    embedding_key='clip_g',
                    embedding_expected_shape=2048,
                    emphasis_name=opts.emphasis,
                    text_projection=True,
                    minimal_clip_skip=2,
                    clip_skip=2,
                    return_pooled=True,
                    final_layer_norm=False,
                    callback_before_encode=set_clip_skip_callback
                )
                engine.is_trainable = False  # for sgm codebase
                engine.legacy_ucg_val = None  # for sgm codebase
                engine.input_key = 'txt'  # for sgm codebase
                conditioner.embedders[i] = engine
                text_cond_models.append(embedder)

        if len(text_cond_models) == 1:
            sd_model.cond_stage_model = text_cond_models[0]
        else:
            sd_model.cond_stage_model = conditioner
    elif type(sd_model.cond_stage_model).__name__ == 'FrozenCLIPEmbedder':  # SD15 Clip
        engine = ClassicTextProcessingEngine(
            text_encoder=forge_objects.clip.cond_stage_model.clip_l,
            tokenizer=forge_objects.clip.tokenizer.clip_l,
            embedding_dir=cmd_opts.embeddings_dir,
            embedding_key='clip_l',
            embedding_expected_shape=768,
            emphasis_name=opts.emphasis,
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=False,
            final_layer_norm=True,
            callback_before_encode=set_clip_skip_callback_and_move_model
        )
        sd_model.cond_stage_model = engine
    elif type(sd_model.cond_stage_model).__name__ == 'FrozenOpenCLIPEmbedder':  # SD21 Clip
        engine = ClassicTextProcessingEngine(
            text_encoder=forge_objects.clip.cond_stage_model.clip_l,
            tokenizer=forge_objects.clip.tokenizer.clip_l,
            embedding_dir=cmd_opts.embeddings_dir,
            embedding_key='clip_l',
            embedding_expected_shape=1024,
            emphasis_name=opts.emphasis,
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=False,
            final_layer_norm=True,
            callback_before_encode=set_clip_skip_callback_and_move_model
        )
        sd_model.cond_stage_model = engine
    else:
        raise NotImplementedError('Bad Clip Class Name:' + type(sd_model.cond_stage_model).__name__)

    timer.record("forge set components")

    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    sd_model.is_sd3 = False
    sd_model.latent_channels = 4
    sd_model.is_sdxl = conditioner is not None
    sd_model.is_sdxl_inpaint = sd_model.is_sdxl and forge_objects.unet.model.diffusion_model.in_channels == 9
    sd_model.is_sd2 = not sd_model.is_sdxl and hasattr(sd_model.cond_stage_model, 'model')
    sd_model.is_sd1 = not sd_model.is_sdxl and not sd_model.is_sd2
    sd_model.is_ssd = sd_model.is_sdxl and 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight' not in sd_model.state_dict().keys()
    if sd_model.is_sdxl:
        extend_sdxl(sd_model)
    sd_model.sd_model_hash = sd_model_hash
    sd_model.sd_model_checkpoint = checkpoint_info.filename
    sd_model.sd_checkpoint_info = checkpoint_info

    @torch.inference_mode()
    def patched_decode_first_stage(x):
        sample = sd_model.forge_objects.vae.first_stage_model.process_out(x)
        sample = sd_model.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)

    @torch.inference_mode()
    def patched_encode_first_stage(x):
        sample = sd_model.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = sd_model.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    sd_model.ema_scope = lambda *args, **kwargs: contextlib.nullcontext()
    sd_model.get_first_stage_encoding = lambda x: x
    sd_model.decode_first_stage = patched_decode_first_stage
    sd_model.encode_first_stage = patched_encode_first_stage
    sd_model.clip = sd_model.cond_stage_model
    sd_model.tiling_enabled = False
    timer.record("forge finalize")

    sd_model.current_lora_hash = str([])
    return sd_model
