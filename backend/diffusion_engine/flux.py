import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.classic_engine import ClassicTextProcessingEngine
from backend.text_processing.t5_engine import T5TextProcessingEngine
from backend.args import dynamic_args
from backend.modules.k_prediction import PredictionFlux
from backend import memory_management


class Flux(ForgeDiffusionEngine):
    matched_guesses = [model_list.Flux, model_list.FluxSchnell]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(
            model_dict={
                'clip_l': huggingface_components['text_encoder'],
                't5xxl': huggingface_components['text_encoder_2']
            },
            tokenizer_dict={
                'clip_l': huggingface_components['tokenizer'],
                't5xxl': huggingface_components['tokenizer_2']
            }
        )

        vae = VAE(model=huggingface_components['vae'])

        if 'schnell' in estimated_config.huggingface_repo.lower():
            k_predictor = PredictionFlux(
                mu=1.0
            )
        else:
            k_predictor = PredictionFlux(
                seq_len=4096,
                base_seq_len=256,
                max_seq_len=4096,
                base_shift=0.5,
                max_shift=1.15,
            )
            self.use_distilled_cfg_scale = True

        unet = UnetPatcher.from_model(
            model=huggingface_components['transformer'],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config
        )

        self.text_processing_engine_l = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_l,
            tokenizer=clip.tokenizer.clip_l,
            embedding_dir=dynamic_args['embedding_dir'],
            embedding_key='clip_l',
            embedding_expected_shape=768,
            emphasis_name=dynamic_args['emphasis_name'],
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=True,
            final_layer_norm=True,
        )

        self.text_processing_engine_t5 = T5TextProcessingEngine(
            text_encoder=clip.cond_stage_model.t5xxl,
            tokenizer=clip.tokenizer.t5xxl,
            emphasis_name=dynamic_args['emphasis_name'],
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    def set_clip_skip(self, clip_skip):
        self.text_processing_engine_l.clip_skip = clip_skip

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        prompt_t5 = []
        prompt_l = []

        for p in prompt:
            if 'SPLIT' in p:
                before_split, after_split = p.split('SPLIT', 1)
                prompt_t5.append(before_split.strip())
                prompt_l.append(after_split.strip())
            else:
                prompt_t5.append(p)
                prompt_l.append(p)

        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        cond_l, pooled_l = self.text_processing_engine_l(prompt_l)
        cond_t5 = self.text_processing_engine_t5(prompt_t5)
        cond = dict(crossattn=cond_t5, vector=pooled_l)

        if self.use_distilled_cfg_scale:
            distilled_cfg_scale = getattr(prompt, 'distilled_cfg_scale', 3.5) or 3.5
            cond['guidance'] = torch.FloatTensor([distilled_cfg_scale] * len(prompt))
            print(f'Distilled CFG Scale: {distilled_cfg_scale}')
        else:
            print('Distilled CFG Scale will be ignored for Schnell')

        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        if 'SPLIT' in prompt:
            prompt_t5, prompt_l = prompt.split('SPLIT', 1)
            prompt_t5 = prompt_t5.strip()
            prompt_l = prompt_l.strip()
        else:
            prompt_t5 = prompt_l = prompt

        t5_token_count = len(self.text_processing_engine_t5.tokenize([prompt_t5])[0])
        t5_max_length = max(255, t5_token_count)

        _, clip_token_count = self.text_processing_engine_l.process_texts([prompt_l])
        clip_max_length = self.text_processing_engine_l.get_target_prompt_token_count(clip_token_count)

        return {
            't5': t5_token_count,
            't5_max': t5_max_length,
            'clip': clip_token_count,
            'clip_max': clip_max_length,
        }

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
