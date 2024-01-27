# Consistent with Kohya/A1111 to reduce differences between model training and inference.

import os
import torch
import ldm_patched.controlnet.cldm
import ldm_patched.k_diffusion.sampling
import ldm_patched.ldm.modules.attention
import ldm_patched.ldm.modules.diffusionmodules.model
import ldm_patched.ldm.modules.diffusionmodules.openaimodel
import ldm_patched.ldm.modules.diffusionmodules.openaimodel
import ldm_patched.modules.args_parser
import ldm_patched.modules.model_base
import ldm_patched.modules.model_management
import ldm_patched.modules.model_patcher
import ldm_patched.modules.samplers
import ldm_patched.modules.sd
import ldm_patched.modules.sd1_clip
import ldm_patched.modules.clip_vision
import ldm_patched.modules.ops as ops

from modules_forge.ops import use_patched_ops
from transformers import CLIPTextModel, CLIPTextConfig, modeling_utils


def patched_SDClipModel__init__(self, max_length=77, freeze=True, layer="last", layer_idx=None,
                                textmodel_json_config=None, dtype=None, special_tokens=None,
                                layer_norm_hidden_state=True, **kwargs):
    torch.nn.Module.__init__(self)
    assert layer in self.LAYERS

    if special_tokens is None:
        special_tokens = {"start": 49406, "end": 49407, "pad": 49407}

    if textmodel_json_config is None:
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(ldm_patched.modules.sd1_clip.__file__)),
                                             "sd1_clip_config.json")

    config = CLIPTextConfig.from_json_file(textmodel_json_config)
    self.num_layers = config.num_hidden_layers

    with use_patched_ops(ops.manual_cast):
        with modeling_utils.no_init_weights():
            self.transformer = CLIPTextModel(config)

    if dtype is not None:
        self.transformer.to(dtype)

    self.transformer.text_model.embeddings.to(torch.float32)

    if freeze:
        self.freeze()

    self.max_length = max_length
    self.layer = layer
    self.layer_idx = None
    self.special_tokens = special_tokens
    self.text_projection = torch.nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
    self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
    self.enable_attention_masks = False

    self.layer_norm_hidden_state = layer_norm_hidden_state
    if layer == "hidden":
        assert layer_idx is not None
        assert abs(layer_idx) < self.num_layers
        self.clip_layer(layer_idx)
    self.layer_default = (self.layer, self.layer_idx)


def patched_SDClipModel_forward(self, tokens):
    backup_embeds = self.transformer.get_input_embeddings()
    device = backup_embeds.weight.device
    tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
    tokens = torch.LongTensor(tokens).to(device)

    attention_mask = None
    if self.enable_attention_masks:
        attention_mask = torch.zeros_like(tokens)
        max_token = self.transformer.get_input_embeddings().weight.shape[0] - 1
        for x in range(attention_mask.shape[0]):
            for y in range(attention_mask.shape[1]):
                attention_mask[x, y] = 1
                if tokens[x, y] == max_token:
                    break

    outputs = self.transformer(input_ids=tokens, attention_mask=attention_mask,
                               output_hidden_states=self.layer == "hidden")
    self.transformer.set_input_embeddings(backup_embeds)

    if self.layer == "last":
        z = outputs.last_hidden_state
    elif self.layer == "pooled":
        z = outputs.pooler_output[:, None, :]
    else:
        z = outputs.hidden_states[self.layer_idx]
        if self.layer_norm_hidden_state:
            z = self.transformer.text_model.final_layer_norm(z)

    if hasattr(outputs, "pooler_output"):
        pooled_output = outputs.pooler_output.float()
    else:
        pooled_output = None

    if self.text_projection is not None and pooled_output is not None:
        pooled_output = pooled_output.float().to(self.text_projection.device) @ self.text_projection.float()

    return z.float(), pooled_output


def patch_all_clip():
    ldm_patched.modules.sd1_clip.SDClipModel.__init__ = patched_SDClipModel__init__
    ldm_patched.modules.sd1_clip.SDClipModel.forward = patched_SDClipModel_forward
    return
