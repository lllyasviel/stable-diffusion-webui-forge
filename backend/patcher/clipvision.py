import torch

from backend.utils import load_torch_file
from backend.state_dict import transformers_convert, state_dict_prefix_replace
from backend import operations, memory_management
from backend.patcher.base import ModelPatcher
from transformers import modeling_utils, CLIPVisionConfig, CLIPVisionModelWithProjection


CLIP_VISION_G = {
  "attention_dropout": 0.0,
  "dropout": 0.0,
  "hidden_act": "gelu",
  "hidden_size": 1664,
  "image_size": 224,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "layer_norm_eps": 1e-05,
  "model_type": "clip_vision_model",
  "num_attention_heads": 16,
  "num_channels": 3,
  "num_hidden_layers": 48,
  "patch_size": 14,
  "projection_dim": 1280,
  "torch_dtype": "float32"
}

CLIP_VISION_H = {
  "attention_dropout": 0.0,
  "dropout": 0.0,
  "hidden_act": "gelu",
  "hidden_size": 1280,
  "image_size": 224,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "intermediate_size": 5120,
  "layer_norm_eps": 1e-05,
  "model_type": "clip_vision_model",
  "num_attention_heads": 16,
  "num_channels": 3,
  "num_hidden_layers": 32,
  "patch_size": 14,
  "projection_dim": 1024,
  "torch_dtype": "float32"
}


CLIP_VISION_VITL = {
  "attention_dropout": 0.0,
  "dropout": 0.0,
  "hidden_act": "quick_gelu",
  "hidden_size": 1024,
  "image_size": 224,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-05,
  "model_type": "clip_vision_model",
  "num_attention_heads": 16,
  "num_channels": 3,
  "num_hidden_layers": 24,
  "patch_size": 14,
  "projection_dim": 768,
  "torch_dtype": "float32"
}


class Output:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, item):
        setattr(self, key, item)


def clip_preprocess(image, size=224):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        scale = (size / min(image.shape[2], image.shape[3]))
        image = torch.nn.functional.interpolate(image, size=(round(scale * image.shape[2]), round(scale * image.shape[3])), mode="bicubic", antialias=True)
        h = (image.shape[2] - size) // 2
        w = (image.shape[3] - size) // 2
        image = image[:, :, h:h + size, w:w + size]
    image = torch.clip((255. * image), 0, 255).round() / 255.0
    return (image - mean.view([3, 1, 1])) / std.view([3, 1, 1])


class ClipVisionModel:
    def __init__(self, config):
        config = CLIPVisionConfig(**config)

        self.load_device = memory_management.text_encoder_device()
        self.offload_device = memory_management.text_encoder_offload_device()

        if memory_management.should_use_fp16(self.load_device, prioritize_performance=False):
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        with operations.using_forge_operations():
            with modeling_utils.no_init_weights():
                self.model = CLIPVisionModelWithProjection(config)

        self.model.to(self.dtype)
        self.patcher = ModelPatcher(
            self.model,
            load_device=self.load_device,
            offload_device=self.offload_device
        )

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def get_sd(self):
        return self.model.state_dict()

    def encode_image(self, image):
        memory_management.load_model_gpu(self.patcher)
        pixel_values = clip_preprocess(image.to(self.load_device))
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

        o = Output()
        o["last_hidden_state"] = outputs.last_hidden_state.to(memory_management.intermediate_device())
        o["penultimate_hidden_states"] = outputs.hidden_states[-2].to(memory_management.intermediate_device())
        o["image_embeds"] = outputs.image_embeds.to(memory_management.intermediate_device())

        return o


def convert_to_transformers(sd, prefix):
    sd_k = sd.keys()
    if "{}transformer.resblocks.0.attn.in_proj_weight".format(prefix) in sd_k:
        keys_to_replace = {
            "{}class_embedding".format(prefix): "vision_model.embeddings.class_embedding",
            "{}conv1.weight".format(prefix): "vision_model.embeddings.patch_embedding.weight",
            "{}positional_embedding".format(prefix): "vision_model.embeddings.position_embedding.weight",
            "{}ln_post.bias".format(prefix): "vision_model.post_layernorm.bias",
            "{}ln_post.weight".format(prefix): "vision_model.post_layernorm.weight",
            "{}ln_pre.bias".format(prefix): "vision_model.pre_layrnorm.bias",
            "{}ln_pre.weight".format(prefix): "vision_model.pre_layrnorm.weight",
        }

        for x in keys_to_replace:
            if x in sd_k:
                sd[keys_to_replace[x]] = sd.pop(x)

        if "{}proj".format(prefix) in sd_k:
            sd['visual_projection.weight'] = sd.pop("{}proj".format(prefix)).transpose(0, 1)

        sd = transformers_convert(sd, prefix, "vision_model.", 48)
    else:
        replace_prefix = {prefix: ""}
        sd = state_dict_prefix_replace(sd, replace_prefix)
    return sd


def load_clipvision_from_sd(sd, prefix="", convert_keys=False):
    if convert_keys:
        sd = convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        config = CLIP_VISION_G
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        config = CLIP_VISION_H
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        config = CLIP_VISION_VITL
    else:
        return None

    clip = ClipVisionModel(config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        print("extra clip vision:", m)
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            t = sd.pop(k)
            del t
    return clip


def load(ckpt_path):
    sd = load_torch_file(ckpt_path)
    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
        return load_clipvision_from_sd(sd, prefix="visual.", convert_keys=True)
    else:
        return load_clipvision_from_sd(sd)
