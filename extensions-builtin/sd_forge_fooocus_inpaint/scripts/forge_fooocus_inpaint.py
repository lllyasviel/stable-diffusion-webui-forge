import os
import torch
import copy

from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from modules_forge.forge_sampler import sampling_prepare
from ldm_patched.modules.utils import load_torch_file
from ldm_patched.modules import model_patcher
from ldm_patched.modules.model_management import cast_to_device, current_loaded_models
from ldm_patched.modules.lora import model_lora_keys_unet


def is_model_loaded(model):
    return any(model == m.model for m in current_loaded_models)


class InpaintHead(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device="cpu"))

    def __call__(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "replicate")
        return torch.nn.functional.conv2d(input=x, weight=self.head)


def load_fooocus_patch(lora: dict, to_load: dict):
    patch_dict = {}
    loaded_keys = set()
    for key in to_load.values():
        if value := lora.get(key, None):
            patch_dict[key] = ("fooocus", value)
            loaded_keys.add(key)

    not_loaded = sum(1 for x in lora if x not in loaded_keys)
    print(f"[Fooocus Patch Loader] {len(loaded_keys)} keys loaded, {not_loaded} remaining keys not found in model.")
    return patch_dict


def calculate_weight_fooocus(weight, alpha, v):
    w1 = cast_to_device(v[0], weight.device, torch.float32)
    if w1.shape == weight.shape:
        w_min = cast_to_device(v[1], weight.device, torch.float32)
        w_max = cast_to_device(v[2], weight.device, torch.float32)
        w1 = (w1 / 255.0) * (w_max - w_min) + w_min
        weight += alpha * cast_to_device(w1, weight.device, weight.dtype)
    else:
        print(f"[Fooocus Patch Loader] weight not merged ({w1.shape} != {weight.shape})")
    return weight


class FooocusInpaintPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if 'diffusion_model.time_embed.0.weight' in state_dict:
            if len(state_dict['diffusion_model.time_embed.0.weight']) == 3:
                return FooocusInpaintPatcher(state_dict)

        return None

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        self.inpaint_head = InpaintHead().to(device=torch.device('cpu'), dtype=torch.float32)
        self.inpaint_head.load_state_dict(load_torch_file(os.path.join(os.path.dirname(__file__), 'fooocus_inpaint_head')))

        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        cond_original = kwargs['cond_original']
        mask_original = kwargs['mask_original']

        unet_original = process.sd_model.forge_objects.unet.clone()
        unet = process.sd_model.forge_objects.unet.clone()
        vae = process.sd_model.forge_objects.vae

        latent_image = vae.encode(cond_original.movedim(1, -1))
        latent_image = process.sd_model.forge_objects.unet.model.latent_format.process_in(latent_image)
        latent_mask = torch.nn.functional.max_pool2d(mask_original, (8, 8)).round().to(cond)
        feed = torch.cat([
            latent_mask.to(device=torch.device('cpu'), dtype=torch.float32),
            latent_image.to(device=torch.device('cpu'), dtype=torch.float32)
        ], dim=1)
        inpaint_head_feature = self.inpaint_head(feed)

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == 0:
                h = h + inpaint_head_feature.to(h)
            return h

        unet.set_model_input_block_patch(input_block_patch)

        lora_keys = model_lora_keys_unet(unet.model, {})
        lora_keys.update({x: x for x in unet.model.state_dict().keys()})
        loaded_lora = load_fooocus_patch(self.state_dict, lora_keys)

        patched = unet.add_patches(loaded_lora, 1.0)

        not_patched_count = sum(1 for x in loaded_lora if x not in patched)

        if not_patched_count > 0:
            print(f"[Fooocus Patch Loader] Failed to load {not_patched_count} keys")

        sigma_start = unet.model.model_sampling.percent_to_sigma(self.start_percent)
        sigma_end = unet.model.model_sampling.percent_to_sigma(self.end_percent)

        def conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
            if timestep > sigma_start or timestep < sigma_end:
                target_model = unet_original
                model_options = copy.deepcopy(model_options)
                if 'transformer_options' in model_options:
                    if 'patches' in model_options['transformer_options']:
                        if 'input_block_patch' in model_options['transformer_options']['patches']:
                            del model_options['transformer_options']['patches']['input_block_patch']
            else:
                target_model = unet

            if not is_model_loaded(target_model):
                sampling_prepare(target_model, x)

            return target_model.model, x, timestep, uncond, cond, cond_scale, model_options, seed

        unet.add_conditioning_modifier(conditioning_modifier)

        process.sd_model.forge_objects.unet = unet
        return


model_patcher.extra_weight_calculators['fooocus'] = calculate_weight_fooocus
add_supported_control_model(FooocusInpaintPatcher)
