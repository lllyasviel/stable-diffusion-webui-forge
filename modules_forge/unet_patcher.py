import copy
import torch

from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel, timestep_embedding, forward_timestep_embed, apply_control
from ldm_patched.modules.model_patcher import ModelPatcher


class UnetPatcher(ModelPatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controlnet_linked_list = None

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
        return n

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

    def add_conditioning_modifier(self, modifier, ensure_uniqueness=False):
        self.append_model_option('conditioning_modifiers', modifier, ensure_uniqueness)
        return

    def add_block_modifier(self, modifier, ensure_uniqueness=False):
        self.append_model_option('block_modifiers', modifier, ensure_uniqueness)
        return


def forge_unet_forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
    transformer_options["original_shape"] = list(x.shape)
    transformer_options["transformer_index"] = 0
    transformer_patches = transformer_options.get("patches", {})

    num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
    image_only_indicator = kwargs.get("image_only_indicator", self.default_image_only_indicator)
    time_context = kwargs.get("time_context", None)

    assert (y is not None) == (
            self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x
    for id, module in enumerate(self.input_blocks):
        transformer_options["block"] = ("input", id)
        h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context,
                                   num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = apply_control(h, control, 'input')
        if "input_block_patch" in transformer_patches:
            patch = transformer_patches["input_block_patch"]
            for p in patch:
                h = p(h, transformer_options)

        hs.append(h)
        if "input_block_patch_after_skip" in transformer_patches:
            patch = transformer_patches["input_block_patch_after_skip"]
            for p in patch:
                h = p(h, transformer_options)

    transformer_options["block"] = ("middle", 0)
    h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options, time_context=time_context,
                               num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
    h = apply_control(h, control, 'middle')

    for id, module in enumerate(self.output_blocks):
        transformer_options["block"] = ("output", id)
        hsp = hs.pop()
        hsp = apply_control(hsp, control, 'output')

        if "output_block_patch" in transformer_patches:
            patch = transformer_patches["output_block_patch"]
            for p in patch:
                h, hsp = p(h, hsp, transformer_options)

        h = torch.cat([h, hsp], dim=1)
        del hsp
        if len(hs) > 0:
            output_shape = hs[-1].shape
        else:
            output_shape = None
        h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape,
                                   time_context=time_context, num_video_frames=num_video_frames,
                                   image_only_indicator=image_only_indicator)

    h = h.type(x.dtype)

    if self.predict_codebook_ids:
        h = self.id_predictor(h)
    else:
        h = self.out(h)

    return h


def patch_all():
    UNetModel.forward = forge_unet_forward
