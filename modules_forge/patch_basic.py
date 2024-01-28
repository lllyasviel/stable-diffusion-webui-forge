import torch

from ldm_patched.modules.controlnet import ControlBase
from ldm_patched.modules.model_patcher import ModelPatcher


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


def patch_all_basics():
    ModelPatcher.__init__ = patched_model_patcher_init
    ModelPatcher.clone = patched_model_patcher_clone
    ModelPatcher.add_patched_controlnet = model_patcher_add_patched_controlnet
    ModelPatcher.list_controlnets = model_patcher_list_controlnets
    ControlBase.control_merge = patched_control_merge
    return
