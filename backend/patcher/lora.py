import torch
import time

import packages_3rdparty.webui_lora_collection.lora as lora_utils_webui
import packages_3rdparty.comfyui_lora_collection.lora as lora_utils_comfyui

from tqdm import tqdm
from backend import memory_management, utils
from backend.args import dynamic_args


class ForgeLoraCollection:
    # TODO
    pass


extra_weight_calculators = {}

lora_utils_forge = ForgeLoraCollection()

lora_collection_priority = [lora_utils_forge, lora_utils_webui, lora_utils_comfyui]


def get_function(function_name: str):
    for lora_collection in lora_collection_priority:
        if hasattr(lora_collection, function_name):
            return getattr(lora_collection, function_name)


def load_lora(lora, to_load):
    patch_dict, remaining_dict = get_function('load_lora')(lora, to_load)
    return patch_dict, remaining_dict


def model_lora_keys_clip(model, key_map={}):
    return get_function('model_lora_keys_clip')(model, key_map)


def model_lora_keys_unet(model, key_map={}):
    return get_function('model_lora_keys_unet')(model, key_map)


@torch.inference_mode()
def weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype):
    # Modified from https://github.com/comfyanonymous/ComfyUI/blob/39f114c44bb99d4a221e8da451d4f2a20119c674/comfy/model_patcher.py#L33

    dora_scale = memory_management.cast_to_device(dora_scale, weight.device, computation_dtype)
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


@torch.inference_mode()
def merge_lora_to_weight(patches, weight, key="online_lora", computation_dtype=torch.float32):
    # Modified from https://github.com/comfyanonymous/ComfyUI/blob/39f114c44bb99d4a221e8da451d4f2a20119c674/comfy/model_patcher.py#L446

    weight_original_dtype = weight.dtype
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
                lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
                if dora_scale is not None:
                    weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype))
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
                    weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype))
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
                    weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype))
                else:
                    weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
            except Exception as e:
                print("ERROR {} {} {}".format(patch_type, key, e))
                raise e
        elif patch_type == "glora":
            if v[4] is not None:
                alpha = v[4] / v[0].shape[0]
            else:
                alpha = 1.0

            dora_scale = v[5]

            a1 = memory_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, computation_dtype)
            a2 = memory_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, computation_dtype)
            b1 = memory_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, computation_dtype)
            b2 = memory_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, computation_dtype)

            try:
                lora_diff = (torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)).reshape(weight.shape)
                if dora_scale is not None:
                    weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype))
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

    weight = weight.to(dtype=weight_original_dtype)
    return weight


from backend import operations


class LoraLoader:
    def __init__(self, model):
        self.model = model
        self.patches = {}
        self.backup = {}
        self.online_backup = []
        self.dirty = False

    def clear_patches(self):
        self.patches.clear()
        self.dirty = True
        return

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
                current_patches.append([strength_patch, patches[k], strength_model, offset, function])
                self.patches[key] = current_patches

        self.dirty = True
        return list(p)

    @torch.inference_mode()
    def refresh(self, target_device=None, offload_device=torch.device('cpu')):
        if not self.dirty:
            return

        self.dirty = False

        execution_start_time = time.perf_counter()

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

        online_mode = dynamic_args.get('online_lora', False)

        # Patch

        for key, current_patches in (tqdm(self.patches.items(), desc=f'Patching LoRAs for {type(self.model).__name__}') if len(self.patches) > 0 else self.patches):
            try:
                parent_layer, child_key, weight = utils.get_attr_with_parent(self.model, key)
                assert isinstance(weight, torch.nn.Parameter)
            except:
                raise ValueError(f"Wrong LoRA Key: {key}")

            if key not in self.backup:
                self.backup[key] = weight.to(device=offload_device)

            if online_mode:
                if not hasattr(parent_layer, 'forge_online_loras'):
                    parent_layer.forge_online_loras = {}

                parent_layer.forge_online_loras[child_key] = current_patches
                self.online_backup.append(parent_layer)
                continue

            bnb_layer = None

            if operations.bnb_avaliable:
                if hasattr(weight, 'bnb_quantized'):
                    bnb_layer = parent_layer
                    if weight.bnb_quantized:
                        weight_original_device = weight.device

                        if target_device is not None:
                            assert target_device.type == 'cuda', 'BNB Must use CUDA!'
                            weight = weight.to(target_device)
                        else:
                            weight = weight.cuda()

                        from backend.operations_bnb import functional_dequantize_4bit
                        weight = functional_dequantize_4bit(weight)

                        if target_device is None:
                            weight = weight.to(device=weight_original_device)
                    else:
                        weight = weight.data

            if target_device is not None:
                try:
                    weight = weight.to(device=target_device)
                except:
                    print('Moving layer weight failed. Retrying by offloading models.')
                    self.model.to(device=offload_device)
                    memory_management.soft_empty_cache()
                    weight = weight.to(device=target_device)

            gguf_cls, gguf_type, gguf_real_shape = None, None, None

            if hasattr(weight, 'is_gguf'):
                from backend.operations_gguf import dequantize_tensor
                gguf_cls = weight.gguf_cls
                gguf_type = weight.gguf_type
                gguf_real_shape = weight.gguf_real_shape
                weight = dequantize_tensor(weight)

            try:
                weight = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)
            except:
                print('Patching LoRA weights failed. Retrying by offloading models.')
                self.model.to(device=offload_device)
                memory_management.soft_empty_cache()
                weight = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)

            if bnb_layer is not None:
                bnb_layer.reload_weight(weight)
                continue

            if gguf_cls is not None:
                from backend.operations_gguf import ParameterGGUF
                weight = gguf_cls.quantize_pytorch(weight, gguf_real_shape)
                utils.set_attr_raw(self.model, key, ParameterGGUF.make(
                    data=weight,
                    gguf_type=gguf_type,
                    gguf_cls=gguf_cls,
                    gguf_real_shape=gguf_real_shape
                ))
                continue

            utils.set_attr_raw(self.model, key, torch.nn.Parameter(weight, requires_grad=False))

        # Time

        moving_time = time.perf_counter() - execution_start_time

        if moving_time > 0.1:
            print(f'LoRA patching has taken {moving_time:.2f} seconds')

        return
