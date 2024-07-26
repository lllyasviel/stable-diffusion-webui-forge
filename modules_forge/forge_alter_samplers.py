from modules import sd_samplers_kdiffusion, sd_samplers_common
from ldm_patched.k_diffusion import sampling as k_diffusion_sampling


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name):
        self.sampler_name = sampler_name
        self.unet = sd_model.forge_objects.unet
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
        super().__init__(sampler_function, sd_model, None)


def build_constructor(sampler_name):
    def constructor(m):
        return AlterSampler(m, sampler_name)

    return constructor


samplers_data_alter = [
    sd_samplers_common.SamplerData('DDPM', build_constructor(sampler_name='ddpm'), ['ddpm'], {}),
]
