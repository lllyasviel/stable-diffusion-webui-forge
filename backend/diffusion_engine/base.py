import torch
import safetensors.torch as sf

from backend import utils


class ForgeObjects:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet = unet
        self.clip = clip
        self.vae = vae
        self.clipvision = clipvision

    def shallow_copy(self):
        return ForgeObjects(
            self.unet,
            self.clip,
            self.vae,
            self.clipvision
        )


class ForgeDiffusionEngine:
    matched_guesses = []

    def __init__(self, estimated_config, huggingface_components):
        self.model_config = estimated_config
        self.is_inpaint = estimated_config.inpaint_model()

        self.forge_objects = None
        self.forge_objects_original = None
        self.forge_objects_after_applying_lora = None

        self.current_lora_hash = str([])

        self.fix_for_webui_backward_compatibility()

    def set_clip_skip(self, clip_skip):
        pass

    def get_first_stage_encoding(self, x):
        return x  # legacy code, do not change

    def get_learned_conditioning(self, prompt: list[str]):
        pass

    def encode_first_stage(self, x):
        pass

    def decode_first_stage(self, x):
        pass

    def get_prompt_lengths_on_ui(self, prompt):
        return {
            't5': 0,
            't5_max': 255,
            'clip': 0,
            'clip_max': 75,
        }

    def is_webui_legacy_model(self):
        return self.is_sd1 or self.is_sd2 or self.is_sdxl or self.is_sd3

    def fix_for_webui_backward_compatibility(self):
        self.tiling_enabled = False
        self.first_stage_model = None
        self.cond_stage_model = None
        self.use_distilled_cfg_scale = False
        self.is_sd1 = False
        self.is_sd2 = False
        self.is_sdxl = False
        self.is_sd3 = False
        return

    def save_unet(self, filename):
        sd = utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model)
        sf.save_file(sd, filename)
        return filename

    def save_checkpoint(self, filename):
        sd = {}
        sd.update(
            utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model, prefix='model.diffusion_model.')
        )
        sd.update(
            utils.get_state_dict_after_quant(self.forge_objects.clip.cond_stage_model, prefix='text_encoders.')
        )
        sd.update(
            utils.get_state_dict_after_quant(self.forge_objects.vae.first_stage_model, prefix='vae.')
        )
        sf.save_file(sd, filename)
        return filename
