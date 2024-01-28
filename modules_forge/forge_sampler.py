import torch
from ldm_patched.modules.conds import CONDRegular, CONDCrossAttn
from ldm_patched.modules.samplers import sampling_function


def cond_from_a1111_to_patched_ldm(cond):
    if isinstance(cond, torch.Tensor):
        result = dict(
            cross_attn=cond,
            model_conds=dict(
                c_crossattn=CONDCrossAttn(cond),
            )
        )
        return [result, ]

    cross_attn = cond['crossattn']
    pooled_output = cond['vector']

    result = dict(
        cross_attn=cross_attn,
        pooled_output=pooled_output,
        model_conds=dict(
            c_crossattn=CONDCrossAttn(cross_attn),
            y=CONDRegular(pooled_output)
        )
    )

    return [result, ]


def cond_from_a1111_to_patched_ldm_weighted(cond, weights):
    transposed = list(map(list, zip(*weights)))
    results = []

    for cond_pre in transposed:
        current_indices = []
        current_weight = 0
        for i, w in cond_pre:
            current_indices.append(i)
            current_weight = w

        if hasattr(cond, 'advanced_indexing'):
            feed = cond.advanced_indexing(current_indices)
        else:
            feed = cond[current_indices]

        h = cond_from_a1111_to_patched_ldm(feed)
        h[0]['strength'] = current_weight
        results += h

    return results


def forge_sample(self, denoiser_params, cond_scale, cond_composition):
    model = self.inner_model.inner_model.forge_objects.unet.model
    x = denoiser_params.x
    timestep = denoiser_params.sigma
    uncond = cond_from_a1111_to_patched_ldm(denoiser_params.text_uncond)
    cond = cond_from_a1111_to_patched_ldm_weighted(denoiser_params.text_cond, cond_composition)
    model_options = self.inner_model.inner_model.forge_objects.unet.model_options
    seed = self.p.seeds[0]

    image_cond_in = denoiser_params.image_cond
    if isinstance(image_cond_in, torch.Tensor):
        if image_cond_in.shape[0] == x.shape[0] \
                and image_cond_in.shape[2] == x.shape[2] \
                and image_cond_in.shape[3] == x.shape[3]:
            uncond[0]['model_conds']['c_concat'] = CONDRegular(image_cond_in)
            cond[0]['model_conds']['c_concat'] = CONDRegular(image_cond_in)

    denoised = sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
    return denoised


# def prepare_sampling(unet, )
