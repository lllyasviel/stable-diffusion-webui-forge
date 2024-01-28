import torch
import ldm_patched.modules.samplers

from ldm_patched.modules.controlnet import ControlBase
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.samplers import get_area_and_mult, can_concat_cond, cond_cat
from ldm_patched.modules import model_management


og_model_patcher_init = ModelPatcher.__init__
og_model_patcher_clone = ModelPatcher.clone


def patched_model_patcher_init(self, *args, **kwargs):
    h = og_model_patcher_init(self, *args, **kwargs)
    self.controlnet_linked_list = None
    return h


def patched_model_patcher_clone(self):
    cloned = og_model_patcher_clone(self)
    cloned.controlnet_linked_list = self.controlnet_linked_list
    return cloned


def model_patcher_add_patched_controlnet(self, cnet):
    cnet.set_previous_controlnet(self.controlnet_linked_list)
    self.controlnet_linked_list = cnet
    return


def model_patcher_list_controlnets(self):
    results = []
    pointer = self.controlnet_linked_list
    while pointer is not None:
        results.append(pointer)
        pointer = pointer.previous_controlnet
    return results


def patched_control_merge(self, control_input, control_output, control_prev, output_dtype):
    out = {'input': [], 'middle': [], 'output': []}

    if control_input is not None:
        for i in range(len(control_input)):
            key = 'input'
            x = control_input[i]
            if x is not None:
                x *= self.strength
                if x.dtype != output_dtype:
                    x = x.to(output_dtype)
            out[key].insert(0, x)

    if control_output is not None:
        for i in range(len(control_output)):
            if i == (len(control_output) - 1):
                key = 'middle'
                index = 0
            else:
                key = 'output'
                index = i
            x = control_output[i]
            if x is not None:
                if self.global_average_pooling:
                    x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

                x *= self.strength
                if x.dtype != output_dtype:
                    x = x.to(output_dtype)

            out[key].append(x)

    if self.positive_advanced_weighting is not None or self.negative_advanced_weighting:
        # TODO: Implement here
        cond_or_uncond = self.current_cond_or_uncond
        a = 0
        pass

    if control_prev is not None:
        for x in ['input', 'middle', 'output']:
            o = out[x]
            for i in range(len(control_prev[x])):
                prev_val = control_prev[x][i]
                if i >= len(o):
                    o.append(prev_val)
                elif prev_val is not None:
                    if o[i] is None:
                        o[i] = prev_val
                    else:
                        if o[i].shape[0] < prev_val.shape[0]:
                            o[i] = prev_val + o[i]
                        else:
                            o[i] += prev_val
    return out


def patched_calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
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

        free_memory = model_management.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp)//i]
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

        if control is not None:
            control.current_cond_or_uncond = cond_or_uncond
            c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond))

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

        c['transformer_options'] = transformer_options

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
        del input_x

        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
        del mult

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond


def patch_all_basics():
    ModelPatcher.__init__ = patched_model_patcher_init
    ModelPatcher.clone = patched_model_patcher_clone
    ModelPatcher.add_patched_controlnet = model_patcher_add_patched_controlnet
    ModelPatcher.list_controlnets = model_patcher_list_controlnets
    ControlBase.control_merge = patched_control_merge
    ldm_patched.modules.samplers.calc_cond_uncond_batch = patched_calc_cond_uncond_batch
    return
