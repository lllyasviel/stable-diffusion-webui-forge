import torch

from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from ldm_patched.modules.samplers import sampling_function
import ldm_patched.ldm.modules.attention as attention


def sdp(q, k, v, transformer_options):
    if q.shape[0] == 0:
        return q

    return attention.optimized_attention(q, k, v, heads=transformer_options["n_heads"], mask=None)


def adain(x, target_std, target_mean):
    if x.shape[0] == 0:
        return x

    std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True, correction=0)
    return (((x - mean) / std) * target_std) + target_mean


def zero_cat(a, b, dim):
    if a.shape[0] == 0:
        return b
    if b.shape[0] == 0:
        return a
    return torch.cat([a, b], dim=dim)


class PreprocessorReference(Preprocessor):
    def __init__(self, name, use_attn=True, use_adain=True, priority=0):
        super().__init__()
        self.name = name
        self.use_attn = use_attn
        self.use_adain = use_adain
        self.sorting_priority = priority
        self.tags = ['Reference']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.slider_1 = PreprocessorParameter(label='Style Fidelity', value=0.5, minimum=0.0, maximum=1.0, step=0.01, visible=True)
        self.show_control_mode = False
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = False
        self.do_not_need_model = True

        self.is_recording_style = False
        self.recorded_attn1 = {}
        self.recorded_h = {}

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unit = kwargs['unit']
        weight = float(unit.weight)
        style_fidelity = float(unit.threshold_a)
        start_percent = float(unit.guidance_start)
        end_percent = float(unit.guidance_end)

        if process.sd_model.is_sdxl:
            style_fidelity = style_fidelity ** 3.0  # sdxl is very sensitive to reference so we lower the weights

        vae = process.sd_model.forge_objects.vae
        # This is a powerful VAE with integrated memory management, bf16, and tiled fallback.

        latent_image = vae.encode(cond.movedim(1, -1))
        latent_image = process.sd_model.forge_objects.unet.model.latent_format.process_in(latent_image)

        gen_seed = process.seeds[0] + 1
        gen_cpu = torch.Generator().manual_seed(gen_seed)

        unet = process.sd_model.forge_objects.unet.clone()
        sigma_max = unet.model.model_sampling.percent_to_sigma(start_percent)
        sigma_min = unet.model.model_sampling.percent_to_sigma(end_percent)

        self.recorded_attn1 = {}
        self.recorded_h = {}

        def conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
            sigma = timestep[0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return model, x, timestep, uncond, cond, cond_scale, model_options, seed

            self.is_recording_style = True

            xt = latent_image.to(x) + torch.randn(x.size(), dtype=x.dtype, generator=gen_cpu).to(x) * sigma
            sampling_function(model, xt, timestep, uncond, cond, 1, model_options, seed)

            self.is_recording_style = False

            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        def block_proc(h, flag, transformer_options):
            if not self.use_adain:
                return h

            if flag != 'after':
                return h

            location = transformer_options['block']

            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return h

            channel = int(h.shape[1])
            minimal_channel = 1500 - 1000 * weight

            if channel < minimal_channel:
                return h

            if self.is_recording_style:
                self.recorded_h[location] = torch.std_mean(h, dim=(2, 3), keepdim=True, correction=0)
                return h
            else:
                cond_indices = transformer_options['cond_indices']
                uncond_indices = transformer_options['uncond_indices']
                cond_or_uncond = transformer_options['cond_or_uncond']
                r_std, r_mean = self.recorded_h[location]

                h_c = h[cond_indices]
                h_uc = h[uncond_indices]

                o_c = adain(h_c, r_std, r_mean)
                o_uc_strong = h_uc
                o_uc_weak = adain(h_uc, r_std, r_mean)
                o_uc = o_uc_weak + (o_uc_strong - o_uc_weak) * style_fidelity

                recon = []
                for cx in cond_or_uncond:
                    if cx == 0:
                        recon.append(o_c)
                    else:
                        recon.append(o_uc)

                o = torch.cat(recon, dim=0)
                return o

        def attn1_proc(q, k, v, transformer_options):
            if not self.use_attn:
                return sdp(q, k, v, transformer_options)

            sigma = transformer_options["sigmas"][0].item()
            if not (sigma_min <= sigma <= sigma_max):
                return sdp(q, k, v, transformer_options)

            location = (transformer_options['block'][0], transformer_options['block'][1],
                        transformer_options['block_index'])

            channel = int(q.shape[2])
            minimal_channel = 1500 - 1280 * weight

            if channel < minimal_channel:
                return sdp(q, k, v, transformer_options)

            if self.is_recording_style:
                self.recorded_attn1[location] = (k, v)
                return sdp(q, k, v, transformer_options)
            else:
                cond_indices = transformer_options['cond_indices']
                uncond_indices = transformer_options['uncond_indices']
                cond_or_uncond = transformer_options['cond_or_uncond']

                q_c = q[cond_indices]
                q_uc = q[uncond_indices]

                k_c = k[cond_indices]
                k_uc = k[uncond_indices]

                v_c = v[cond_indices]
                v_uc = v[uncond_indices]

                k_r, v_r = self.recorded_attn1[location]

                o_c = sdp(q_c, zero_cat(k_c, k_r, dim=1), zero_cat(v_c, v_r, dim=1), transformer_options)
                o_uc_strong = sdp(q_uc, k_uc, v_uc, transformer_options)
                o_uc_weak = sdp(q_uc, zero_cat(k_uc, k_r, dim=1), zero_cat(v_uc, v_r, dim=1), transformer_options)
                o_uc = o_uc_weak + (o_uc_strong - o_uc_weak) * style_fidelity

                recon = []
                for cx in cond_or_uncond:
                    if cx == 0:
                        recon.append(o_c)
                    else:
                        recon.append(o_uc)

                o = torch.cat(recon, dim=0)
                return o

        unet.add_block_modifier(block_proc)
        unet.add_conditioning_modifier(conditioning_modifier)
        unet.set_model_replace_all(attn1_proc, 'attn1')

        process.sd_model.forge_objects.unet = unet

        return cond, mask


add_supported_preprocessor(PreprocessorReference(
    name='reference_only',
    use_attn=True,
    use_adain=False,
    priority=100
))

add_supported_preprocessor(PreprocessorReference(
    name='reference_adain',
    use_attn=False,
    use_adain=True
))

add_supported_preprocessor(PreprocessorReference(
    name='reference_adain+attn',
    use_attn=True,
    use_adain=True
))
