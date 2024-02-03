import copy

from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.sample import convert_cond
from ldm_patched.modules.samplers import encode_model_conds


class UnetPatcher(ModelPatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controlnet_linked_list = None
        self.extra_preserved_memory_during_sampling = 0
        self.extra_model_patchers_during_sampling = []

    def clone(self):
        n = UnetPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device,
                        weight_inplace_update=self.weight_inplace_update)

        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        n.controlnet_linked_list = self.controlnet_linked_list
        n.extra_preserved_memory_during_sampling = self.extra_preserved_memory_during_sampling
        n.extra_model_patchers_during_sampling = self.extra_model_patchers_during_sampling.copy()
        return n

    def add_extra_preserved_memory_during_sampling(self, memory_in_bytes: int):
        # Use this to ask Forge to preserve a certain amount of memory during sampling.
        # If GPU VRAM is 8 GB, and memory_in_bytes is 2GB, i.e., memory_in_bytes = 2 * 1024 * 1024 * 1024
        # Then the sampling will always use less than 6GB memory by dynamically offload modules to CPU RAM.
        # You can estimate this using model_management.module_size(any_pytorch_model) to get size of any pytorch models.
        self.extra_preserved_memory_during_sampling += memory_in_bytes
        return

    def add_extra_model_patcher_during_sampling(self, model_patcher: ModelPatcher):
        # Use this to ask Forge to move extra model patchers to GPU during sampling.
        # This method will manage GPU memory perfectly.
        self.extra_model_patchers_during_sampling.append(model_patcher)
        return

    def add_patched_controlnet(self, cnet):
        cnet.set_previous_controlnet(self.controlnet_linked_list)
        self.controlnet_linked_list = cnet
        return

    def list_controlnets(self):
        results = []
        pointer = self.controlnet_linked_list
        while pointer is not None:
            results.append(pointer)
            pointer = pointer.previous_controlnet
        return results

    def append_model_option(self, k, v, ensure_uniqueness=False):
        if k not in self.model_options:
            self.model_options[k] = []

        if ensure_uniqueness and v in self.model_options[k]:
            return

        self.model_options[k].append(v)
        return

    def append_transformer_option(self, k, v, ensure_uniqueness=False):
        if 'transformer_options' not in self.model_options:
            self.model_options['transformer_options'] = {}

        to = self.model_options['transformer_options']

        if k not in to:
            to[k] = []

        if ensure_uniqueness and v in to[k]:
            return

        to[k].append(v)
        return

    def add_conditioning_modifier(self, modifier, ensure_uniqueness=False):
        self.append_model_option('conditioning_modifiers', modifier, ensure_uniqueness)
        return

    def add_block_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option('block_modifiers', modifier, ensure_uniqueness)
        return

    def add_block_inner_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option('block_inner_modifiers', modifier, ensure_uniqueness)
        return

    def add_controlnet_conditioning_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option('controlnet_conditioning_modifiers', modifier, ensure_uniqueness)
        return

    def set_model_replace_all(self, patch, target="attn1"):
        for block_name in ['input', 'middle', 'output']:
            for number in range(16):
                for transformer_index in range(16):
                    self.set_model_patch_replace(patch, target, block_name, number, transformer_index)
        return

    def encode_conds_after_clip(self, conds, noise, prompt_type="positive"):
        return encode_model_conds(
            model_function=self.model.extra_conds,
            conds=convert_cond(conds),
            noise=noise,
            device=noise.device,
            prompt_type=prompt_type
        )
