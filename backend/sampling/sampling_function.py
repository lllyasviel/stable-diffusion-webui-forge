# Started from some codes from early ComfyUI and then 80% rewritten,
# mainly for supporting different special control methods in Forge
# Copyright Forge 2024


import torch
import math
import collections

from backend import memory_management
from backend.sampling.condition import Condition, compile_conditions, compile_weighted_conditions
from backend.operations import cleanup_cache
from backend.args import dynamic_args, args
from backend import utils


def get_area_and_mult(conds, x_in, timestep_in):
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    if 'area' in conds:
        area = conds['area']
    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]

    if 'mask' in conds:
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds['mask']
        assert (mask.shape[1] == x_in.shape[2])
        assert (mask.shape[2] == x_in.shape[3])
        mask = mask[:, area[2]:area[0] + area[2], area[3]:area[1] + area[3]] * mask_strength
        mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
    else:
        mask = torch.ones_like(input_x)
    mult = mask * strength

    if 'mask' not in conds:
        rr = 8
        if area[2] != 0:
            for t in range(rr):
                mult[:, :, t:1 + t, :] *= ((1.0 / rr) * (t + 1))
        if (area[0] + area[2]) < x_in.shape[2]:
            for t in range(rr):
                mult[:, :, area[0] - 1 - t:area[0] - t, :] *= ((1.0 / rr) * (t + 1))
        if area[3] != 0:
            for t in range(rr):
                mult[:, :, :, t:1 + t] *= ((1.0 / rr) * (t + 1))
        if (area[1] + area[3]) < x_in.shape[3]:
            for t in range(rr):
                mult[:, :, :, area[1] - 1 - t:area[1] - t] *= ((1.0 / rr) * (t + 1))

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

    control = conds.get('control', None)

    patches = None
    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches'])
    return cond_obj(input_x, mult, conditioning, area, control, patches)


def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True


def can_concat_cond(c1, c2):
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list):
    c_crossattn = []
    c_concat = []
    c_adm = []
    crossattn_max_len = 0

    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out


def compute_cond_mark(cond_or_uncond, sigmas):
    cond_or_uncond_size = int(sigmas.shape[0])

    cond_mark = []
    for cx in cond_or_uncond:
        cond_mark += [cx] * cond_or_uncond_size

    cond_mark = torch.Tensor(cond_mark).to(sigmas)
    return cond_mark


def compute_cond_indices(cond_or_uncond, sigmas):
    cl = int(sigmas.shape[0])

    cond_indices = []
    uncond_indices = []
    for i, cx in enumerate(cond_or_uncond):
        if cx == 0:
            cond_indices += list(range(i * cl, (i + 1) * cl))
        else:
            uncond_indices += list(range(i * cl, (i + 1) * cl))

    return cond_indices, uncond_indices


def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1

    to_run = []
    for x in cond:
        p = get_area_and_mult(x, x_in, timestep)
        if p is None:
            continue

        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        if memory_management.signal_empty_cache:
            memory_management.soft_empty_cache()

        free_memory = memory_management.get_free_memory(x_in.device)

        if (not args.disable_gpu_warning) and x_in.device.type == 'cuda':
            free_memory_mb = free_memory / (1024.0 * 1024.0)
            safe_memory_mb = 1536.0
            if free_memory_mb < safe_memory_mb:
                print(f"\n\n----------------------")
                print(f"[Low GPU VRAM Warning] Your current GPU free memory is {free_memory_mb:.2f} MB for this diffusion iteration.")
                print(f"[Low GPU VRAM Warning] This number is lower than the safe value of {safe_memory_mb:.2f} MB.")
                print(f"[Low GPU VRAM Warning] If you continue, you may cause NVIDIA GPU performance degradation for this diffusion process, and the speed may be extremely slow (about 10x slower).")
                print(f"[Low GPU VRAM Warning] To solve the problem, you can set the 'GPU Weights' (on the top of page) to a lower value.")
                print(f"[Low GPU VRAM Warning] If you cannot find 'GPU Weights', you can click the 'all' option in the 'UI' area on the left-top corner of the webpage.")
                print(f"[Low GPU VRAM Warning] If you want to take the risk of NVIDIA GPU fallback and test the 10x slower speed, you can (but are highly not recommended to) add '--disable-gpu-warning' to CMD flags to remove this warning.")
                print(f"----------------------\n\n")

        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        transformer_options["cond_mark"] = compute_cond_mark(cond_or_uncond=cond_or_uncond, sigmas=timestep)
        transformer_options["cond_indices"], transformer_options["uncond_indices"] = compute_cond_indices(cond_or_uncond=cond_or_uncond, sigmas=timestep)

        c['transformer_options'] = transformer_options

        if control is not None:
            p = control
            while p is not None:
                p.transformer_options = transformer_options
                p = p.previous_controlnet
            control_cond = c.copy()  # get_control may change items in this dict, so we need to copy it
            c['control'] = control.get_control(input_x, timestep_, control_cond, len(cond_or_uncond))
            c['control_model'] = control

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
        del input_x

        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                out_cond[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:, :, area[o][2]:area[o][0] + area[o][2], area[o][3]:area[o][1] + area[o][3]] += mult[o]
        del mult

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond


def sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None, return_full=False):
    edit_strength = sum((item['strength'] if 'strength' in item else 1) for item in cond)

    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    for fn in model_options.get("sampler_pre_cfg_function", []):
        model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)

    cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)

    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    elif not math.isclose(edit_strength, 1.0):
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    if return_full:
        return cfg_result, cond_pred, uncond_pred

    return cfg_result


def sampling_function(self, denoiser_params, cond_scale, cond_composition):
    unet_patcher = self.inner_model.inner_model.forge_objects.unet
    model = unet_patcher.model
    control = unet_patcher.controlnet_linked_list
    extra_concat_condition = unet_patcher.extra_concat_condition
    x = denoiser_params.x
    timestep = denoiser_params.sigma
    uncond = compile_conditions(denoiser_params.text_uncond)
    cond = compile_weighted_conditions(denoiser_params.text_cond, cond_composition)
    model_options = unet_patcher.model_options
    seed = self.p.seeds[0]

    if extra_concat_condition is not None:
        image_cond_in = extra_concat_condition
    else:
        image_cond_in = denoiser_params.image_cond

    if isinstance(image_cond_in, torch.Tensor):
        if image_cond_in.shape[0] == x.shape[0] \
                and image_cond_in.shape[2] == x.shape[2] \
                and image_cond_in.shape[3] == x.shape[3]:
            if uncond is not None:
                for i in range(len(uncond)):
                    uncond[i]['model_conds']['c_concat'] = Condition(image_cond_in)
            for i in range(len(cond)):
                cond[i]['model_conds']['c_concat'] = Condition(image_cond_in)

    if control is not None:
        for h in cond:
            h['control'] = control
        if uncond is not None:
            for h in uncond:
                h['control'] = control

    for modifier in model_options.get('conditioning_modifiers', []):
        model, x, timestep, uncond, cond, cond_scale, model_options, seed = modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

    denoised, cond_pred, uncond_pred = sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options, seed, return_full=True)
    return denoised, cond_pred, uncond_pred


def sampling_prepare(unet, x):
    B, C, H, W = x.shape

    memory_estimation_function = unet.model_options.get('memory_peak_estimation_modifier', unet.memory_required)

    unet_inference_memory = memory_estimation_function([B * 2, C, H, W])
    additional_inference_memory = unet.extra_preserved_memory_during_sampling
    additional_model_patchers = unet.extra_model_patchers_during_sampling

    if unet.controlnet_linked_list is not None:
        additional_inference_memory += unet.controlnet_linked_list.inference_memory_requirements(unet.model_dtype())
        additional_model_patchers += unet.controlnet_linked_list.get_models()

    if unet.has_online_lora():
        lora_memory = utils.nested_compute_size(unet.lora_patches, element_size=utils.dtype_to_element_size(unet.model.computation_dtype))
        additional_inference_memory += lora_memory

    memory_management.load_models_gpu(
        models=[unet] + additional_model_patchers,
        memory_required=unet_inference_memory,
        hard_memory_preservation=additional_inference_memory
    )

    if unet.has_online_lora():
        utils.nested_move_to_device(unet.lora_patches, device=unet.current_device, dtype=unet.model.computation_dtype)

    real_model = unet.model

    percent_to_timestep_function = lambda p: real_model.predictor.percent_to_sigma(p)

    for cnet in unet.list_controlnets():
        cnet.pre_run(real_model, percent_to_timestep_function)

    return


def sampling_cleanup(unet):
    if unet.has_online_lora():
        utils.nested_move_to_device(unet.lora_patches, device=unet.offload_device)
    for cnet in unet.list_controlnets():
        cnet.cleanup()
    cleanup_cache()
    return
