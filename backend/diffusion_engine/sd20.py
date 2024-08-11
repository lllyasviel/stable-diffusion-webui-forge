import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.classic_engine import ClassicTextProcessingEngine
from backend.args import dynamic_args
from backend import memory_management


class StableDiffusion2(ForgeDiffusionEngine):
    matched_guesses = [model_list.SD20]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(
            model_dict={
                'clip_h': huggingface_components['text_encoder']
            },
            tokenizer_dict={
                'clip_h': huggingface_components['tokenizer']
            }
        )

        vae = VAE(model=huggingface_components['vae'])

        unet = UnetPatcher.from_model(
            model=huggingface_components['unet'],
            diffusers_scheduler=huggingface_components['scheduler'],
            config=estimated_config
        )

        self.text_processing_engine = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_h,
            tokenizer=clip.tokenizer.clip_h,
            embedding_dir=dynamic_args['embedding_dir'],
            embedding_key='clip_h',
            embedding_expected_shape=1024,
            emphasis_name=dynamic_args['emphasis_name'],
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=False,
            final_layer_norm=True,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # WebUI Legacy
        self.is_sd2 = True

    def set_clip_skip(self, clip_skip):
        self.text_processing_engine.clip_skip = clip_skip

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        cond = self.text_processing_engine(prompt)
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        _, token_count = self.text_processing_engine.process_texts([prompt])
        return token_count, self.text_processing_engine.get_target_prompt_token_count(token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
