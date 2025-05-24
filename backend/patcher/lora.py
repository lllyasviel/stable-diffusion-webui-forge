import torch

import packages_3rdparty.webui_lora_collection.lora as lora_utils_webui
import packages_3rdparty.comfyui_lora_collection.lora as lora_utils_comfyui

from backend import memory_management, utils


extra_weight_calculators = {}
lora_collection_priority = [lora_utils_webui, lora_utils_comfyui]


def get_function(function_name: str):
    for lora_collection in lora_collection_priority:
        if hasattr(lora_collection, function_name):
            return getattr(lora_collection, function_name)


def load_lora(lora, to_load):
    patch_dict, remaining_dict = get_function('load_lora')(lora, to_load)
    return patch_dict, remaining_dict


def inner_str(k, prefix="", suffix=""):
    return k[len(prefix):-len(suffix)]


def model_lora_keys_clip(model, key_map={}):
    model_keys, key_maps = get_function('model_lora_keys_clip')(model, key_map)

    for model_key in model_keys:
        if model_key.endswith(".weight"):
            if model_key.startswith("t5xxl.transformer."):
                for prefix in ['te1', 'te2', 'te3']:
                    formatted = inner_str(model_key, "t5xxl.transformer.", ".weight")
                    formatted = formatted.replace(".", "_")
                    formatted = f"lora_{prefix}_{formatted}"
                    key_map[formatted] = model_key

    return key_maps


def model_lora_keys_unet(model, key_map={}):
    model_keys, key_maps = get_function('model_lora_keys_unet')(model, key_map)

    # TODO: OFT

    return key_maps


@torch.inference_mode()
def weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function):
    # Modified from https://github.com/comfyanonymous/ComfyUI/blob/80a44b97f5cbcb890896e2b9e65d177f1ac6a588/comfy/weight_adapter/base.py#L42
    dora_scale = memory_management.cast_to_device(dora_scale, weight.device, computation_dtype)
    lora_diff *= alpha
    weight_calc = weight + function(lora_diff).type(weight.dtype)

    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]
    if wd_on_output_axis:
        weight_norm = (
            weight.reshape(weight.shape[0], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[0], *[1] * (weight.dim() - 1))
        )
    else:
        weight_norm = (
            weight_calc.transpose(0, 1)
            .reshape(weight_calc.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
            .transpose(0, 1)
        )
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * (weight_calc)
    else:
        weight[:] = weight_calc
    return weight


@torch.inference_mode()
def merge_lora_to_weight(patches, weight, key="online_lora", computation_dtype=torch.float32):
    # Modified from https://github.com/comfyanonymous/ComfyUI/blob/39f114c44bb99d4a221e8da451d4f2a20119c674/comfy/model_patcher.py#L446

    weight_dtype_backup = None

    if computation_dtype == weight.dtype:
        weight = weight.clone()
    else:
        weight_dtype_backup = weight.dtype
        weight = weight.to(dtype=computation_dtype)

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
            v = (merge_lora_to_weight(v[1:], v[0].clone(), key),)

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

        elif patch_type == "set":
            weight.copy_(v[0])

        elif patch_type == "lora":
            mat1 = memory_management.cast_to_device(v[0], weight.device, computation_dtype)
            mat2 = memory_management.cast_to_device(v[1], weight.device, computation_dtype)
            dora_scale = v[4]

            if v[2] is not None:
                alpha = v[2] / mat2.shape[0]
            else:
                alpha = 1.0

            if v[3] is not None:
                mat3 = memory_management.cast_to_device(v[3], weight.device, computation_dtype)
                final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
            
            try:
                lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))
                
                try:
                    lora_diff = lora_diff.reshape(weight.shape)
                except:
                    if weight.shape[1] < lora_diff.shape[1]:
                        expand_factor = (lora_diff.shape[1] - weight.shape[1])
                        weight = torch.nn.functional.pad(weight, (0, expand_factor), mode='constant', value=0)                        
                    elif weight.shape[1] > lora_diff.shape[1]:
                        # expand factor should be 1*64 (for FluxTools Canny or Depth), or 5*64 (for FluxTools Fill)
                        expand_factor = (weight.shape[1] - lora_diff.shape[1])
                        lora_diff = torch.nn.functional.pad(lora_diff, (0, expand_factor), mode='constant', value=0)
                    
                if dora_scale is not None:
                    weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function)
                else:
                    weight += function(((strength * alpha) * lora_diff).type(weight.dtype))

            except Exception as e:
                print("ERROR {} {} {}".format(patch_type, key, e))
                raise e
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
                w1 = torch.mm(memory_management.cast_to_device(w1_a, weight.device, computation_dtype),
                              memory_management.cast_to_device(w1_b, weight.device, computation_dtype))
            else:
                w1 = memory_management.cast_to_device(w1, weight.device, computation_dtype)

            if w2 is None:
                dim = w2_b.shape[0]
                if t2 is None:
                    w2 = torch.mm(memory_management.cast_to_device(w2_a, weight.device, computation_dtype),
                                  memory_management.cast_to_device(w2_b, weight.device, computation_dtype))
                else:
                    w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      memory_management.cast_to_device(t2, weight.device, computation_dtype),
                                      memory_management.cast_to_device(w2_b, weight.device, computation_dtype),
                                      memory_management.cast_to_device(w2_a, weight.device, computation_dtype))
            else:
                w2 = memory_management.cast_to_device(w2, weight.device, computation_dtype)

            if len(w2.shape) == 4:
                w1 = w1.unsqueeze(2).unsqueeze(2)
            if v[2] is not None and dim is not None:
                alpha = v[2] / dim
            else:
                alpha = 1.0

            try:
                lora_diff = torch.kron(w1, w2).reshape(weight.shape)
                if dora_scale is not None:
                    weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function)
                else:
                    weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
            except Exception as e:
                print("ERROR {} {} {}".format(patch_type, key, e))
                raise e
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
                                  memory_management.cast_to_device(t1, weight.device, computation_dtype),
                                  memory_management.cast_to_device(w1b, weight.device, computation_dtype),
                                  memory_management.cast_to_device(w1a, weight.device, computation_dtype))

                m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                  memory_management.cast_to_device(t2, weight.device, computation_dtype),
                                  memory_management.cast_to_device(w2b, weight.device, computation_dtype),
                                  memory_management.cast_to_device(w2a, weight.device, computation_dtype))
            else:
                m1 = torch.mm(memory_management.cast_to_device(w1a, weight.device, computation_dtype),
                              memory_management.cast_to_device(w1b, weight.device, computation_dtype))
                m2 = torch.mm(memory_management.cast_to_device(w2a, weight.device, computation_dtype),
                              memory_management.cast_to_device(w2b, weight.device, computation_dtype))

            try:
                lora_diff = (m1 * m2).reshape(weight.shape)
                if dora_scale is not None:
                    weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function)
                else:
                    weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
            except Exception as e:
                print("ERROR {} {} {}".format(patch_type, key, e))
                raise e

        elif patch_type == "glora":
            dora_scale = v[5]
            
            old_glora = False
            if v[3].shape[1] == v[2].shape[0] == v[0].shape[0] == v[1].shape[1]:
                old_glora = True

            if v[3].shape[0] == v[2].shape[1] == v[0].shape[1] == v[1].shape[0]:
                if old_glora and v[1].shape[0] == weight.shape[0] and weight.shape[0] == weight.shape[1]:
                    pass
                else:
                    old_glora = False

            a1 = memory_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, computation_dtype)
            a2 = memory_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, computation_dtype)
            b1 = memory_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, computation_dtype)
            b2 = memory_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, computation_dtype)

            if v[4] is None:
                alpha = 1.0
            else:
                if old_glora:
                    alpha = v[4] / v[0].shape[0]
                else:
                    alpha = v[4] / v[1].shape[0]

            try:
                if old_glora:
                    lora_diff = (torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1).to(dtype=computation_dtype), a2), a1)).reshape(weight.shape) #old lycoris glora
                else:
                    if weight.dim() > 2:
                        lora_diff = torch.einsum("o i ..., i j -> o j ...", torch.einsum("o i ..., i j -> o j ...", weight.to(dtype=computation_dtype), a1), a2).reshape(weight.shape)
                    else:
                        lora_diff = torch.mm(torch.mm(weight.to(dtype=computation_dtype), a1), a2).reshape(weight.shape)
                    lora_diff += torch.mm(b1, b2).reshape(weight.shape)

                if dora_scale is not None:
                    weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function)
                else:
                    weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
            except Exception as e:
                print("ERROR {} {} {}".format(patch_type, key, e))
                raise e
        elif patch_type in extra_weight_calculators:
            weight = extra_weight_calculators[patch_type](weight, strength, v)
        else:
            print("patch type not recognized {} {}".format(patch_type, key))

        if old_weight is not None:
            weight = old_weight

    if weight_dtype_backup is not None:
        weight = weight.to(dtype=weight_dtype_backup)

    return weight


def get_parameter_devices(model):
    parameter_devices = {}
    for key, p in model.named_parameters():
        parameter_devices[key] = p.device
    return parameter_devices


def set_parameter_devices(model, parameter_devices):
    for key, device in parameter_devices.items():
        p = utils.get_attr(model, key)
        if p.device != device:
            p = utils.tensor2parameter(p.to(device=device))
            utils.set_attr_raw(model, key, p)
    return model


from backend import operations


class LoraLoader:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.online_backup = []
        self.loaded_hash = str([])

    @torch.inference_mode()
    def refresh(self, lora_patches, offload_device=torch.device('cpu'), force_refresh=False):
        hashes = str(list(lora_patches.keys()))

        if hashes == self.loaded_hash and not force_refresh:
            return

        # Merge Patches

        all_patches = {}

        for (_, _, _, online_mode), patches in lora_patches.items():
            for key, current_patches in patches.items():
                all_patches[(key, online_mode)] = all_patches.get((key, online_mode), []) + current_patches

        # Initialize

        memory_management.signal_empty_cache = True

        parameter_devices = get_parameter_devices(self.model)

        # Restore

        for m in set(self.online_backup):
            del m.forge_online_loras

        self.online_backup = []

        for k, w in self.backup.items():
            if not isinstance(w, torch.nn.Parameter):
                # In very few cases
                w = torch.nn.Parameter(w, requires_grad=False)

            utils.set_attr_raw(self.model, k, w)

        self.backup = {}

        set_parameter_devices(self.model, parameter_devices=parameter_devices)

        # Patch

        for (key, online_mode), current_patches in all_patches.items():
            try:
                parent_layer, child_key, weight = utils.get_attr_with_parent(self.model, key)
                assert isinstance(weight, torch.nn.Parameter)
            except:
                raise ValueError(f"Wrong LoRA Key: {key}")

            if online_mode:
                if not hasattr(parent_layer, 'forge_online_loras'):
                    parent_layer.forge_online_loras = {}

                parent_layer.forge_online_loras[child_key] = current_patches
                self.online_backup.append(parent_layer)
                continue

            if key not in self.backup:
                self.backup[key] = weight.to(device=offload_device)

            bnb_layer = None

            if hasattr(weight, 'bnb_quantized') and operations.bnb_avaliable:
                bnb_layer = parent_layer
                from backend.operations_bnb import functional_dequantize_4bit
                weight = functional_dequantize_4bit(weight)

            gguf_cls = getattr(weight, 'gguf_cls', None)
            gguf_parameter = None

            if gguf_cls is not None:
                gguf_parameter = weight
                from backend.operations_gguf import dequantize_tensor
                weight = dequantize_tensor(weight)

            try:
                weight = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)
            except:
                print('Patching LoRA weights out of memory. Retrying by offloading models.')
                set_parameter_devices(self.model, parameter_devices={k: offload_device for k in parameter_devices.keys()})
                memory_management.soft_empty_cache()
                weight = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)

            if bnb_layer is not None:
                bnb_layer.reload_weight(weight)
                continue

            if gguf_cls is not None:
                gguf_cls.quantize_pytorch(weight, gguf_parameter)
                continue

            utils.set_attr_raw(self.model, key, torch.nn.Parameter(weight, requires_grad=False))

        # End

        set_parameter_devices(self.model, parameter_devices=parameter_devices)
        self.loaded_hash = hashes
        return
