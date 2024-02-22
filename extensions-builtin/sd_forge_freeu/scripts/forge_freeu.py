import gradio as gr

from modules import scripts
from ldm_patched.contrib.external_freelunch import FreeU_V2


opFreeU_V2 = FreeU_V2()


# def Fourier_filter(x, threshold, scale):
#     x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
#     x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
#     B, C, H, W = x_freq.shape
#     mask = torch.ones((B, C, H, W), device=x.device)
#     crow, ccol = H // 2, W //2
#     mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
#     x_freq = x_freq * mask
#     x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
#     x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
#     return x_filtered.to(x.dtype)
#
#
# def set_freeu_v2_patch(model, b1, b2, s1, s2):
#     model_channels = model.model.model_config.unet_config["model_channels"]
#     scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
#
#     def output_block_patch(h, hsp, *args, **kwargs):
#         scale = scale_dict.get(h.shape[1], None)
#         if scale is not None:
#             hidden_mean = h.mean(1).unsqueeze(1)
#             B = hidden_mean.shape[0]
#             hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
#             hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
#             hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / \
#                           (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
#             h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)
#             hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
#         return h, hsp
#
#     m = model.clone()
#     m.set_model_output_block_patch(output_block_patch)
#     return m


class FreeUForForge(scripts.Script):
    sorting_priority = 12

    def title(self):
        return "FreeU Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            freeu_enabled = gr.Checkbox(label='Enabled', value=False)
            freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
            freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
            freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
            freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)

        return freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2 = script_args

        if not freeu_enabled:
            return

        unet = p.sd_model.forge_objects.unet

        # unet = set_freeu_v2_patch(unet, freeu_b1, freeu_b2, freeu_s1, freeu_s2)
        unet = opFreeU_V2.patch(unet, freeu_b1, freeu_b2, freeu_s1, freeu_s2)[0]

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            freeu_enabled=freeu_enabled,
            freeu_b1=freeu_b1,
            freeu_b2=freeu_b2,
            freeu_s1=freeu_s1,
            freeu_s2=freeu_s2,
        ))

        return
