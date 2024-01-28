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


def patch_all_basics():
    ModelPatcher.__init__ = patched_model_patcher_init
    ModelPatcher.clone = patched_model_patcher_clone
    ModelPatcher.add_patched_controlnet = model_patcher_add_patched_controlnet
    ModelPatcher.list_controlnets = model_patcher_list_controlnets
    return
