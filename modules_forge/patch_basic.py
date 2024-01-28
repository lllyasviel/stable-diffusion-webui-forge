from ldm_patched.modules.model_patcher import ModelPatcher


og_model_patcher_init = ModelPatcher.__init__
og_model_patcher_clone = ModelPatcher.clone


def patched_model_patcher_init(self, *args, **kwargs):
    h = og_model_patcher_init(self, *args, **kwargs)
    self.control_options = []
    return h


def patched_model_patcher_clone(self):
    cloned = og_model_patcher_clone(self)
    cloned.control_options = [x for x in self.control_options]
    return cloned


def patch_all_basics():
    ModelPatcher.__init__ = patched_model_patcher_init
    ModelPatcher.clone = patched_model_patcher_clone
    return
