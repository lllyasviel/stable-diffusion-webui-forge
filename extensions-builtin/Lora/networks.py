import os
import re

import lora_patches
import network

import torch
from typing import Union

from modules import shared, sd_models, errors, scripts
from ldm_patched.modules.utils import load_torch_file
from ldm_patched.modules.sd import load_lora_for_models


lora_state_dict_cache = {}
lora_state_dict_cache_max_length = 5

module_types = []


re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    return key


def assign_network_names_to_compvis_modules(sd_model):
    pass


def load_network(name, network_on_disk):
    pass


def purge_networks_from_memory():
    pass


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    global lora_state_dict_cache

    current_sd = sd_models.model_data.get_sd_model()
    if current_sd is None:
        return

    networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()
        networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]

    compiled_lora_targets = []
    for a, b, c in zip(networks_on_disk, unet_multipliers, te_multipliers):
        compiled_lora_targets.append([a.filename, b, c])

    compiled_lora_targets_hash = str(compiled_lora_targets)

    if current_sd.current_lora_hash == compiled_lora_targets_hash:
        return

    current_sd.current_lora_hash = compiled_lora_targets_hash
    current_sd.forge_objects.unet = current_sd.forge_objects.unet_original
    current_sd.forge_objects.clip = current_sd.forge_objects.clip_original

    for filename, strength_model, strength_clip in compiled_lora_targets:
        if filename in lora_state_dict_cache:
            lora_sd = lora_state_dict_cache[filename]
        else:
            if len(lora_state_dict_cache) > lora_state_dict_cache_max_length:
                lora_state_dict_cache = {}

            lora_sd = load_torch_file(filename, safe_load=True)
            lora_state_dict_cache[filename] = lora_sd

        current_sd.forge_objects.unet, current_sd.forge_objects.clip = load_lora_for_models(
            current_sd.forge_objects.unet, current_sd.forge_objects.clip, lora_sd, strength_model, strength_clip)

    return


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    pass


def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    pass


def network_forward(org_module, input, original_forward):
    pass


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    pass


def network_Linear_forward(self, input):
    pass


def network_Linear_load_state_dict(self, *args, **kwargs):
    pass


def network_Conv2d_forward(self, input):
    pass


def network_Conv2d_load_state_dict(self, *args, **kwargs):
    pass


def network_GroupNorm_forward(self, input):
    pass


def network_GroupNorm_load_state_dict(self, *args, **kwargs):
    pass


def network_LayerNorm_forward(self, input):
    pass


def network_LayerNorm_load_state_dict(self, *args, **kwargs):
    pass


def network_MultiheadAttention_forward(self, *args, **kwargs):
    pass


def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    pass


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    candidates = list(shared.walk_files(shared.cmd_opts.lora_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    candidates += list(shared.walk_files(shared.cmd_opts.lyco_dir_backcompat, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    for filename in candidates:
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params):
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything

    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


originals: lora_patches.LoraPatches = None

extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()
