import torch
from modules import sd_samplers_kdiffusion, sd_samplers_common

from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
from ldm_patched.modules.samplers import calculate_sigmas_scheduler


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name, scheduler_name):
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.unet = sd_model.forge_objects.unet

        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
        super().__init__(sampler_function, sd_model, None)

    def get_sigmas(self, p, steps):
        if self.scheduler_name == 'turbo':
            timesteps = torch.flip(torch.arange(1, steps + 1) * float(1000.0 / steps) - 1, (0,)).round().long().clip(0, 999)
            sigmas = self.unet.model.model_sampling.sigma(timesteps)
            sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        else:
            sigmas = calculate_sigmas_scheduler(self.unet.model, self.scheduler_name, steps)
        return sigmas.to(self.unet.load_device)


def build_constructor(sampler_name, scheduler_name):
    def constructor(m):
        return AlterSampler(m, sampler_name, scheduler_name)

    return constructor


samplers_data_alter = [
    sd_samplers_common.SamplerData('DDPM', build_constructor(sampler_name='ddpm', scheduler_name='normal'), ['ddpm'], {}),
    sd_samplers_common.SamplerData('DDPM Karras', build_constructor(sampler_name='ddpm', scheduler_name='karras'), ['ddpm_karras'], {}),
    sd_samplers_common.SamplerData('Euler A Turbo', build_constructor(sampler_name='euler_ancestral', scheduler_name='turbo'), ['euler_ancestral_turbo'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M Turbo', build_constructor(sampler_name='dpmpp_2m', scheduler_name='turbo'), ['dpmpp_2m_turbo'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M SDE Turbo', build_constructor(sampler_name='dpmpp_2m_sde', scheduler_name='turbo'), ['dpmpp_2m_sde_turbo'], {}),
    sd_samplers_common.SamplerData('LCM Karras', build_constructor(sampler_name='lcm', scheduler_name='karras'), ['lcm_karras'], {}),
    sd_samplers_common.SamplerData('Euler SGMUniform', build_constructor(sampler_name='euler', scheduler_name='sgm_uniform'), ['euler_sgm_uniform'], {}),
    sd_samplers_common.SamplerData('Euler A SGMUniform', build_constructor(sampler_name='euler_ancestral', scheduler_name='sgm_uniform'), ['euler_ancestral_sgm_uniform'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M SGMUniform', build_constructor(sampler_name='dpmpp_2m', scheduler_name='sgm_uniform'), ['dpmpp_2m_sgm_uniform'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M SDE SGMUniform', build_constructor(sampler_name='dpmpp_2m_sde', scheduler_name='sgm_uniform'), ['dpmpp_2m_sde_sgm_uniform'], {}),
]
