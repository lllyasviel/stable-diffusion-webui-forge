import collections
import importlib
import os
import sys
import math
import threading
import enum

import torch
import re
import safetensors.torch
from omegaconf import OmegaConf, ListConfig
from urllib import request
import gc
import contextlib

from modules import paths, shared, modelloader, devices, script_callbacks, sd_vae, sd_disable_initialization, errors, hashes, sd_models_config, sd_unet, sd_models_xl, cache, extra_networks, processing, lowvram, sd_hijack, patches
from modules.shared import opts, cmd_opts
from modules.timer import Timer
import numpy as np
from backend.loader import forge_loader
from backend import memory_management
from backend.args import dynamic_args
from backend.utils import load_torch_file


model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))

checkpoints_list = {}
checkpoint_aliases = {}
checkpoint_alisases = checkpoint_aliases  # for compatibility with old name
checkpoints_loaded = collections.OrderedDict()


class ModelType(enum.Enum):
    SD1 = 1
    SD2 = 2
    SDXL = 3
    SSD = 4
    SD3 = 5


def replace_key(d, key, new_key, value):
    keys = list(d.keys())

    d[new_key] = value

    if key not in keys:
        return d

    index = keys.index(key)
    keys[index] = new_key

    new_d = {k: d[k] for k in keys}

    d.clear()
    d.update(new_d)
    return d


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)
        abs_ckpt_dir = os.path.abspath(shared.cmd_opts.ckpt_dir) if shared.cmd_opts.ckpt_dir is not None else None

        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        if abs_ckpt_dir and abspath.startswith(abs_ckpt_dir):
            name = abspath.replace(abs_ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        def read_metadata():
            metadata = read_metadata_from_safetensors(filename)
            self.modelspec_thumbnail = metadata.pop('modelspec.thumbnail', None)

            return metadata

        self.metadata = {}
        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "checkpoint/" + name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading metadata for {filename}")

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{name}")
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'
        self.short_title = self.name_for_extra if self.shorthash is None else f'{self.name_for_extra} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, self.name_for_extra, f'{name} [{self.hash}]']
        if self.shorthash:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_aliases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return

        shorthash = self.sha256[0:10]
        if self.shorthash == self.sha256[0:10]:
            return self.shorthash

        self.shorthash = shorthash

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

        old_title = self.title
        self.title = f'{self.name} [{self.shorthash}]'
        self.short_title = f'{self.name_for_extra} [{self.shorthash}]'

        replace_key(checkpoints_list, old_title, self.title, self)
        self.register()

        return self.shorthash

    def __str__(self):
        return str(dict(filename=self.filename, hash=self.hash))

    def __repr__(self):
        return str(dict(filename=self.filename, hash=self.hash))


# try:
#     # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
#     from transformers import logging, CLIPModel  # noqa: F401
#
#     logging.set_verbosity_error()
# except Exception:
#     pass


def setup_model():
    """called once at startup to do various one-time tasks related to SD models"""

    os.makedirs(model_path, exist_ok=True)

    enable_midas_autodownload()
    patch_given_betas()


def checkpoint_tiles(use_short=False):
    return [x.short_title if use_short else x.name for x in checkpoints_list.values()]


def list_models():
    checkpoints_list.clear()
    checkpoint_aliases.clear()

    cmd_ckpt = shared.cmd_opts.ckpt

    model_list = modelloader.load_models(model_path=model_path, model_url=None, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors", ".gguf"], download_name=None, ext_blacklist=[".vae.ckpt", ".vae.safetensors"])

    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}", file=sys.stderr)

    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


re_strip_checksum = re.compile(r"\s*\[[^]]+]\s*$")

def match_checkpoint_to_name(name):
    name = name.split(' [')[0]

    for ckptname in checkpoints_list.values():
        title = ckptname.title.split(' [')[0]
        if (name in title) or (title in name):
            return ckptname.short_title if shared.opts.sd_checkpoint_dropdown_use_short else ckptname.name.split(' [')[0]

    return name

def get_closet_checkpoint_match(search_string):
    if not search_string:
        return None

    checkpoint_info = checkpoint_aliases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info

    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    search_string_without_checksum = re.sub(re_strip_checksum, '', search_string)
    found = sorted([info for info in checkpoints_list.values() if search_string_without_checksum in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    return None


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def select_checkpoint():
    """Raises `FileNotFoundError` if no checkpoints are found."""
    model_checkpoint = shared.opts.sd_model_checkpoint

    checkpoint_info = checkpoint_aliases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        print('You do not have any model!')
        return None

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


def transform_checkpoint_dict_key(k, replacements):
    pass


def get_state_dict_from_checkpoint(pl_sd):
    pass


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"

        res = {}

        try:
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v[0:1] == '{':
                    try:
                        res[k] = json.loads(v)
                    except Exception:
                        pass
        except Exception:
             errors.report(f"Error reading metadata from file: {filename}", exc_info=True)

        return res


def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    pass


def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    if checkpoint_info in checkpoints_loaded:
        # use checkpoint cache
        print(f"Loading weights [{sd_model_hash}] from cache")
        # move to end as latest
        checkpoints_loaded.move_to_end(checkpoint_info)
        return checkpoints_loaded[checkpoint_info]

    print(f"Loading weights [{sd_model_hash}] from {checkpoint_info.filename}")
    res = load_torch_file(checkpoint_info.filename)
    timer.record("load weights from disk")

    return res


def SkipWritingToConfig():
    return contextlib.nullcontext()


def check_fp8(model):
    pass


def set_model_type(model, state_dict):
    pass


def set_model_fields(model):
    pass


def load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict, timer):
    pass


def enable_midas_autodownload():
    pass


def patch_given_betas():
    pass


def repair_config(sd_config, state_dict=None):
    pass


def rescale_zero_terminal_snr_abar(alphas_cumprod):
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return alphas_bar


def apply_alpha_schedule_override(sd_model, p=None):
    """
    Applies an override to the alpha schedule of the model according to settings.
    - downcasts the alpha schedule to half precision
    - rescales the alpha schedule to have zero terminal SNR
    """

    if not hasattr(sd_model, 'alphas_cumprod') or not hasattr(sd_model, 'alphas_cumprod_original'):
        return

    sd_model.alphas_cumprod = sd_model.alphas_cumprod_original.to(shared.device)

    if opts.use_downcasted_alpha_bar:
        if p is not None:
            p.extra_generation_params['Downcast alphas_cumprod'] = opts.use_downcasted_alpha_bar
        sd_model.alphas_cumprod = sd_model.alphas_cumprod.half().to(shared.device)

    if opts.sd_noise_schedule == "Zero Terminal SNR":
        if p is not None:
            p.extra_generation_params['Noise Schedule'] = opts.sd_noise_schedule
        sd_model.alphas_cumprod = rescale_zero_terminal_snr_abar(sd_model.alphas_cumprod).to(shared.device)


# This is a dummy class for backward compatibility when model is not load - for extensions like prompt all in one.
class FakeInitialModel:
    def __init__(self):
        self.cond_stage_model = None
        self.chunk_length = 75

    def get_prompt_lengths_on_ui(self, prompt):
        r = len(prompt.strip('!,. ').replace(' ', ',').replace('.', ',').replace('!', ',').replace(',,', ',').replace(',,', ',').replace(',,', ',').replace(',,', ',').split(','))
        return r, math.ceil(max(r, 1) / self.chunk_length) * self.chunk_length


class SdModelData:
    def __init__(self):
        self.sd_model = FakeInitialModel()
        self.forge_loading_parameters = {}
        self.forge_hash = ''

    def get_sd_model(self):
        return self.sd_model

    def set_sd_model(self, v):
        self.sd_model = v


model_data = SdModelData()


def get_empty_cond(sd_model):
    pass


def send_model_to_cpu(m):
    pass


def model_target_device(m):
    return devices.device


def send_model_to_device(m):
    pass


def send_model_to_trash(m):
    pass


def instantiate_from_config(config, state_dict=None):
    pass


def get_obj_from_str(string, reload=False):
    pass


def load_model(checkpoint_info=None, already_loaded_state_dict=None):
    pass


def reuse_model_from_already_loaded(sd_model, checkpoint_info, timer):
    pass


def reload_model_weights(sd_model=None, info=None, forced_reload=False):
    pass


def unload_model_weights(sd_model=None, info=None):
    memory_management.unload_all_models()
    pass


def apply_token_merging(sd_model, token_merging_ratio):
    if token_merging_ratio <= 0:
        return

    print(f'token_merging_ratio = {token_merging_ratio}')

    from backend.misc.tomesd import TomePatcher

    sd_model.forge_objects.unet = TomePatcher().patch(
        model=sd_model.forge_objects.unet,
        ratio=token_merging_ratio
    )

    return


@torch.inference_mode()
def forge_model_reload():
    current_hash = str(model_data.forge_loading_parameters)

    if model_data.forge_hash == current_hash:
        return model_data.sd_model, False

    print('Loading Model: ' + str(model_data.forge_loading_parameters))

    timer = Timer()

    if model_data.sd_model:
        model_data.sd_model = None
        memory_management.unload_all_models()
        memory_management.soft_empty_cache()
        gc.collect()

    timer.record("unload existing model")

    checkpoint_info = model_data.forge_loading_parameters['checkpoint_info']

    if checkpoint_info is None:
        raise ValueError('You do not have any model! Please download at least one model in [models/Stable-diffusion].')

    state_dict = checkpoint_info.filename
    additional_state_dicts = model_data.forge_loading_parameters.get('additional_modules', [])

    timer.record("cache state dict")

    dynamic_args['forge_unet_storage_dtype'] = model_data.forge_loading_parameters.get('unet_storage_dtype', None)
    dynamic_args['embedding_dir'] = cmd_opts.embeddings_dir
    dynamic_args['emphasis_name'] = opts.emphasis
    sd_model = forge_loader(state_dict, additional_state_dicts=additional_state_dicts)
    timer.record("forge model load")

    sd_model.extra_generation_params = {}
    sd_model.comments = []
    sd_model.sd_checkpoint_info = checkpoint_info
    sd_model.filename = checkpoint_info.filename
    sd_model.sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    model_data.set_sd_model(sd_model)

    script_callbacks.model_loaded_callback(sd_model)

    timer.record("scripts callbacks")

    print(f"Model loaded in {timer.summary()}.")

    model_data.forge_hash = current_hash

    return sd_model, True
