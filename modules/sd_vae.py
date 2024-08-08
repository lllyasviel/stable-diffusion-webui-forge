import os
import collections
from dataclasses import dataclass

from modules import paths, shared, devices, script_callbacks, sd_models, extra_networks, lowvram, sd_hijack, hashes

import glob
from copy import deepcopy
from backend.utils import load_torch_file


vae_path = os.path.abspath(os.path.join(paths.models_path, "VAE"))
vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
vae_dict = {}


base_vae = None
loaded_vae_file = None
checkpoint_info = None

checkpoints_loaded = collections.OrderedDict()


def get_loaded_vae_name():
    if loaded_vae_file is None:
        return None

    return os.path.basename(loaded_vae_file)


def get_loaded_vae_hash():
    if loaded_vae_file is None:
        return None

    sha256 = hashes.sha256(loaded_vae_file, 'vae')

    return sha256[0:10] if sha256 else None


def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model):
    global base_vae, checkpoint_info
    if checkpoint_info != model.sd_checkpoint_info:
        assert not loaded_vae_file, "Trying to store non-base VAE!"
        base_vae = deepcopy(model.first_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global loaded_vae_file
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        print("Restoring base VAE")
        _load_vae_dict(model, base_vae)
        loaded_vae_file = None
    delete_base_vae()


def get_filename(filepath):
    return os.path.basename(filepath)


def refresh_vae_list():
    vae_dict.clear()

    paths = [
        os.path.join(sd_models.model_path, '**/*.vae.ckpt'),
        os.path.join(sd_models.model_path, '**/*.vae.pt'),
        os.path.join(sd_models.model_path, '**/*.vae.safetensors'),
        os.path.join(vae_path, '**/*.ckpt'),
        os.path.join(vae_path, '**/*.pt'),
        os.path.join(vae_path, '**/*.safetensors'),
    ]

    if shared.cmd_opts.ckpt_dir is not None and os.path.isdir(shared.cmd_opts.ckpt_dir):
        paths += [
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.ckpt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.pt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.safetensors'),
        ]

    if shared.cmd_opts.vae_dir is not None and os.path.isdir(shared.cmd_opts.vae_dir):
        paths += [
            os.path.join(shared.cmd_opts.vae_dir, '**/*.ckpt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.pt'),
            os.path.join(shared.cmd_opts.vae_dir, '**/*.safetensors'),
        ]

    candidates = []
    for path in paths:
        candidates += glob.iglob(path, recursive=True)

    for filepath in candidates:
        name = get_filename(filepath)
        vae_dict[name] = filepath

    vae_dict.update(dict(sorted(vae_dict.items(), key=lambda item: shared.natural_sort_key(item[0]))))


def find_vae_near_checkpoint(checkpoint_file):
    checkpoint_path = os.path.basename(checkpoint_file).rsplit('.', 1)[0]
    for vae_file in vae_dict.values():
        if os.path.basename(vae_file).startswith(checkpoint_path):
            return vae_file

    return None


@dataclass
class VaeResolution:
    vae: str = None
    source: str = None
    resolved: bool = True

    def tuple(self):
        return self.vae, self.source


def is_automatic():
    return shared.opts.sd_vae in {"Automatic", "auto"}  # "auto" for people with old config


def resolve_vae_from_setting() -> VaeResolution:
    if shared.opts.sd_vae == "None":
        return VaeResolution()

    vae_from_options = vae_dict.get(shared.opts.sd_vae, None)
    if vae_from_options is not None:
        return VaeResolution(vae_from_options, 'specified in settings')

    if not is_automatic():
        print(f"Couldn't find VAE named {shared.opts.sd_vae}; using None instead")

    return VaeResolution(resolved=False)


def resolve_vae_from_user_metadata(checkpoint_file) -> VaeResolution:
    metadata = extra_networks.get_user_metadata(checkpoint_file)
    vae_metadata = metadata.get("vae", None)
    if vae_metadata is not None and vae_metadata != "Automatic":
        if vae_metadata == "None":
            return VaeResolution()

        vae_from_metadata = vae_dict.get(vae_metadata, None)
        if vae_from_metadata is not None:
            return VaeResolution(vae_from_metadata, "from user metadata")

    return VaeResolution(resolved=False)


def resolve_vae_near_checkpoint(checkpoint_file) -> VaeResolution:
    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    if vae_near_checkpoint is not None and (not shared.opts.sd_vae_overrides_per_model_preferences or is_automatic()):
        return VaeResolution(vae_near_checkpoint, 'found near the checkpoint')

    return VaeResolution(resolved=False)


def resolve_vae(checkpoint_file) -> VaeResolution:
    if shared.cmd_opts.vae_path is not None:
        return VaeResolution(shared.cmd_opts.vae_path, 'from commandline argument')

    if shared.opts.sd_vae_overrides_per_model_preferences and not is_automatic():
        return resolve_vae_from_setting()

    res = resolve_vae_from_user_metadata(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_vae_near_checkpoint(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_vae_from_setting()

    return res


def load_vae_dict(filename, map_location):
    pass


def load_vae(model, vae_file=None, vae_source="from unknown source"):
    raise NotImplementedError('Forge does not use this!')


# don't call this from outside
def _load_vae_dict(model, vae_dict_1):
    pass


def clear_loaded_vae():
    pass


unspecified = object()


def reload_vae_weights(sd_model=None, vae_file=unspecified):
    raise NotImplementedError('Forge does not use this!')
