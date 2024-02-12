import torch
import copy

from modules_forge.supported_preprocessor import PreprocessorClipVision, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor


def revision_conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
    revision_conditions = model_options['revision_conditions']
    noise_augmentor = model.noise_augmentor
    noise_augment_merge = 0.0
    ignore_prompt = False

    adm_inputs = []
    weights = []
    noise_aug = []
    for revision_condition in revision_conditions:
        adm_cond = revision_condition['cond'].image_embeds
        weight = revision_condition["weight"]
        noise_augment = revision_condition["noise_aug"]
        noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
        c_adm, noise_level_emb = noise_augmentor(adm_cond.to(x.device),
                                                 noise_level=torch.tensor([noise_level], device=x.device), seed=seed)
        adm_out = torch.cat((c_adm, noise_level_emb), 1) * weight
        weights.append(weight)
        noise_aug.append(noise_augment)
        adm_inputs.append(adm_out)
        if revision_condition["ignore_prompt"]:
            ignore_prompt = True

    if len(noise_aug) > 1:
        adm_out = torch.stack(adm_inputs).sum(0)
        noise_augment = noise_augment_merge
        noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
        c_adm, noise_level_emb = noise_augmentor(adm_out[:, :noise_augmentor.time_embed.dim],
                                                 noise_level=torch.tensor([noise_level], device=x.device))
        adm_out = torch.cat((c_adm, noise_level_emb), 1)

    new_y = adm_out[:, :1280]
    cond = copy.deepcopy(cond)
    uncond = copy.deepcopy(uncond)

    for c in cond:
        c['model_conds']['y'].cond[:, :1280] = new_y.clone()

    for c in uncond:
        c['model_conds']['y'].cond[:, :1280] = torch.zeros_like(new_y)

    if ignore_prompt:
        for c in cond + uncond:
            c['model_conds']['c_crossattn'].cond = torch.zeros_like(c['model_conds']['c_crossattn'].cond)

    return model, x, timestep, uncond, cond, cond_scale, model_options, seed


class PreprocessorClipVisionForRevision(PreprocessorClipVision):
    def __init__(self, name, url, filename, ignore_prompt=False):
        super().__init__(name, url, filename)
        self.tags = ['Revision']
        self.model_filename_filters = ['Revision']
        self.do_not_need_model = True
        self.ignore_prompt = ignore_prompt
        self.slider_1 = PreprocessorParameter(
            label="Noise Augmentation", minimum=0.0, maximum=1.0, value=0.0, visible=True)

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unit = kwargs['unit']

        weight = float(unit.weight)
        noise_aug = float(unit.threshold_a)

        unet = process.sd_model.forge_objects.unet.clone()

        if 'revision_conditions' not in unet.model_options:
            unet.model_options['revision_conditions'] = []

        unet.model_options['revision_conditions'].append(dict(
            cond=cond,
            weight=weight,
            noise_aug=noise_aug,
            ignore_prompt=self.ignore_prompt
        ))

        unet.add_conditioning_modifier(revision_conditioning_modifier, ensure_uniqueness=True)

        process.sd_model.forge_objects.unet = unet

        return cond, mask


add_supported_preprocessor(PreprocessorClipVisionForRevision(
    name='CLIP-G (Revision)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors',
    ignore_prompt=False
))

add_supported_preprocessor(PreprocessorClipVisionForRevision(
    name='CLIP-G (Revision ignore prompt)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors',
    ignore_prompt=True
))
