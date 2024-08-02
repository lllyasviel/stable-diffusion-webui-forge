# Model Patching API Template Extracted From ComfyUI
# The actual implementation for those APIs are from Forge, implemented from scratch (after forge-v1.0.1),
# and may have certain level of differences.

import torch
import copy
import inspect

from backend import memory_management, utils

extra_weight_calculators = {}


def weight_decompose(dora_scale, weight, lora_diff, alpha, strength):
    dora_scale = memory_management.cast_to_device(dora_scale, weight.device, torch.float32)
    lora_diff *= alpha
    weight_calc = weight + lora_diff.type(weight.dtype)
    weight_norm = (
        weight_calc.transpose(0, 1)
        .reshape(weight_calc.shape[1], -1)
        .norm(dim=1, keepdim=True)
        .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
        .transpose(0, 1)
    )

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * weight_calc
    else:
        weight[:] = weight_calc
    return weight


def set_model_options_patch_replace(model_options, patch, name, block_name, number, transformer_index=None):
    to = model_options["transformer_options"].copy()

    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    else:
        to["patches_replace"][name] = to["patches_replace"][name].copy()

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)
    to["patches_replace"][name][block] = patch
    model_options["transformer_options"] = to
    return model_options


def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_post_cfg_function"] = model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


def set_model_options_pre_cfg_function(model_options, pre_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_pre_cfg_function"] = model_options.get("sampler_pre_cfg_function", []) + [pre_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False):
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

        self.weight_inplace_update = weight_inplace_update

    def model_size(self):
        if self.size > 0:
            return self.size
        self.size = memory_management.module_size(self.model)
        return self.size

    def clone(self):
        n = ModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        return n

    def is_clone(self, other):
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"])  # Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_post_cfg_function(self.model_options, post_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_pre_cfg_function(self.model_options, pre_cfg_function, disable_cfg1_optimization)

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_vae_encode_wrapper(self, wrapper_function):
        self.model_options["model_vae_encode_wrapper"] = wrapper_function

    def set_model_vae_decode_wrapper(self, wrapper_function):
        self.model_options["model_vae_decode_wrapper"] = wrapper_function

    def set_model_vae_regulation(self, vae_regulation):
        self.model_options["model_vae_regulation"] = vae_regulation

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(self.model_options, patch, name, block_name, number, transformer_index=transformer_index)

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block_patch")

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        else:
            if name in self.object_patches_backup:
                return self.object_patches_backup[name]
            else:
                return utils.get_attr(self.model, name)

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        model_sd = self.model.state_dict()
        for k in patches:
            offset = None
            function = None
            if isinstance(k, str):
                key = k
            else:
                offset = k[1]
                key = k[0]
                if len(k) > 2:
                    function = k[2]

            if key in model_sd:
                p.add(k)
                current_patches = self.patches.get(key, [])
                current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                self.patches[key] = current_patches

        return list(p)

    def get_key_patches(self, filter_prefix=None):
        memory_management.unload_model_clones(self)
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            if k in self.patches:
                p[k] = [model_sd[k]] + self.patches[k]
            else:
                p[k] = (model_sd[k],)
        return p

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_model(self, device_to=None, patch_weights=True):
        for k in self.object_patches:
            old = utils.get_attr(self.model, k)
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old
            utils.set_attr_raw(self.model, k, self.object_patches[k])

        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    print("could not patch. key doesn't exist in model:", key)
                    continue

                weight = model_sd[key]

                inplace_update = self.weight_inplace_update

                if key not in self.backup:
                    self.backup[key] = weight.to(device=self.offload_device, copy=inplace_update)

                if device_to is not None:
                    temp_weight = memory_management.cast_to_device(weight, device_to, torch.float32, copy=True)
                else:
                    temp_weight = weight.to(torch.float32, copy=True)
                out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
                if inplace_update:
                    utils.copy_to_param(self.model, key, out_weight)
                else:
                    utils.set_attr(self.model, key, out_weight)
                del temp_weight

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            strength = p[0]
            v = p[1]
            strength_model = p[2]
            offset = p[3]
            function = p[4]
            if function is None:
                function = lambda a: a

            old_weight = None
            if offset is not None:
                old_weight = weight
                weight = weight.narrow(offset[0], offset[1], offset[2])

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                v = (self.calculate_weight(v[1:], v[0].clone(), key),)

            patch_type = ''

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                w1 = v[0]
                if strength != 0.0:
                    if w1.shape != weight.shape:
                        if w1.ndim == weight.ndim == 4:
                            new_shape = [max(n, m) for n, m in zip(weight.shape, w1.shape)]
                            print(f'Merged with {key} channel changed to {new_shape}')
                            new_diff = strength * memory_management.cast_to_device(w1, weight.device, weight.dtype)
                            new_weight = torch.zeros(size=new_shape).to(weight)
                            new_weight[:weight.shape[0], :weight.shape[1], :weight.shape[2], :weight.shape[3]] = weight
                            new_weight[:new_diff.shape[0], :new_diff.shape[1], :new_diff.shape[2], :new_diff.shape[3]] += new_diff
                            new_weight = new_weight.contiguous().clone()
                            weight = new_weight
                        else:
                            print("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                    else:
                        weight += strength * memory_management.cast_to_device(w1, weight.device, weight.dtype)
            elif patch_type == "lora":
                mat1 = memory_management.cast_to_device(v[0], weight.device, torch.float32)
                mat2 = memory_management.cast_to_device(v[1], weight.device, torch.float32)
                dora_scale = v[4]
                if v[2] is not None:
                    alpha = v[2] / mat2.shape[0]
                else:
                    alpha = 1.0

                if v[3] is not None:
                    mat3 = memory_management.cast_to_device(v[3], weight.device, torch.float32)
                    final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                    mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
                try:
                    lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength))
                    else:
                        weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
                except Exception as e:
                    print("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "lokr":
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dora_scale = v[8]
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(memory_management.cast_to_device(w1_a, weight.device, torch.float32),
                                  memory_management.cast_to_device(w1_b, weight.device, torch.float32))
                else:
                    w1 = memory_management.cast_to_device(w1, weight.device, torch.float32)

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(memory_management.cast_to_device(w2_a, weight.device, torch.float32),
                                      memory_management.cast_to_device(w2_b, weight.device, torch.float32))
                    else:
                        w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                          memory_management.cast_to_device(t2, weight.device, torch.float32),
                                          memory_management.cast_to_device(w2_b, weight.device, torch.float32),
                                          memory_management.cast_to_device(w2_a, weight.device, torch.float32))
                else:
                    w2 = memory_management.cast_to_device(w2, weight.device, torch.float32)

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha = v[2] / dim
                else:
                    alpha = 1.0

                try:
                    lora_diff = torch.kron(w1, w2).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength))
                    else:
                        weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
                except Exception as e:
                    print("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "loha":
                w1a = v[0]
                w1b = v[1]
                if v[2] is not None:
                    alpha = v[2] / w1b.shape[0]
                else:
                    alpha = 1.0

                w2a = v[3]
                w2b = v[4]
                dora_scale = v[7]
                if v[5] is not None:
                    t1 = v[5]
                    t2 = v[6]
                    m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      memory_management.cast_to_device(t1, weight.device, torch.float32),
                                      memory_management.cast_to_device(w1b, weight.device, torch.float32),
                                      memory_management.cast_to_device(w1a, weight.device, torch.float32))

                    m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      memory_management.cast_to_device(t2, weight.device, torch.float32),
                                      memory_management.cast_to_device(w2b, weight.device, torch.float32),
                                      memory_management.cast_to_device(w2a, weight.device, torch.float32))
                else:
                    m1 = torch.mm(memory_management.cast_to_device(w1a, weight.device, torch.float32),
                                  memory_management.cast_to_device(w1b, weight.device, torch.float32))
                    m2 = torch.mm(memory_management.cast_to_device(w2a, weight.device, torch.float32),
                                  memory_management.cast_to_device(w2b, weight.device, torch.float32))

                try:
                    lora_diff = (m1 * m2).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength))
                    else:
                        weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
                except Exception as e:
                    print("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "glora":
                if v[4] is not None:
                    alpha = v[4] / v[0].shape[0]
                else:
                    alpha = 1.0

                dora_scale = v[5]

                a1 = memory_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, torch.float32)
                a2 = memory_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, torch.float32)
                b1 = memory_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, torch.float32)
                b2 = memory_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, torch.float32)

                try:
                    lora_diff = (torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength))
                    else:
                        weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
                except Exception as e:
                    print("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type in extra_weight_calculators:
                weight = extra_weight_calculators[patch_type](weight, strength, v)
            else:
                print("patch type not recognized {} {}".format(patch_type, key))

            if old_weight is not None:
                weight = old_weight

        return weight

    def unpatch_model(self, device_to=None):
        keys = list(self.backup.keys())

        if self.weight_inplace_update:
            for k in keys:
                utils.copy_to_param(self.model, k, self.backup[k])
        else:
            for k in keys:
                utils.set_attr(self.model, k, self.backup[k])

        self.backup = {}

        if device_to is not None:
            self.model.to(device_to)
            self.current_device = device_to

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            utils.set_attr_raw(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup = {}
