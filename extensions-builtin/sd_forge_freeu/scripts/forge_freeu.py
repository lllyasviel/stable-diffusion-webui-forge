import torch
import gradio as gr

from modules import scripts


def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


def patch_freeu_v2(unet_patcher, b1, b2, s1, s2):
    model_channels = unet_patcher.model.diffusion_model.config["model_channels"]
    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        scale = scale_dict.get(h.shape[1], None)
        if scale is not None:
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

            h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)

            if hsp.device not in on_cpu_devices:
                try:
                    hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                except:
                    print("Device", hsp.device, "does not support the torch.fft functions used in the FreeU node, switching to CPU.")
                    on_cpu_devices[hsp.device] = True
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
            else:
                hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

        return h, hsp

    m = unet_patcher.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


class FreeUForForge(scripts.Script):
    sorting_priority = 12  # It will be the 12th item on UI.

    def title(self):
        return "FreeU Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title(),
                          elem_id="extensions-freeu",
                          elem_classes=["extensions-freeu"]):
            with gr.Row(equal_height=True):
                freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                freeu_preset = gr.Dropdown(['sd1.4', 'sd1.5', 'sd2.1', 'sdxl'], value='sd1.5', type='value', label='Presets', allow_custom_value=True)
            with gr.Row():
                freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.5)
                freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.6)
            with gr.Row():
                freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.9)
                freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.2)

        self.infotext_fields = [
            (freeu_enabled, lambda d: d.get("freeu_enabled", False)),
            # note: 'freeu_preset is NOT written to infotext, loaded values always considered to be custom
            (freeu_preset, lambda d: d.get("freeu_preset", 'custom')),
            (freeu_b1, "freeu_b1"),
            (freeu_b2, "freeu_b2"),
            (freeu_s1, "freeu_s1"),
            (freeu_s2, "freeu_s2"),
        ]

        def setPresetText ():
            return 'custom'

        def setPresetValues (preset):
            # these values from: https://github.com/ChenyangSi/FreeU
            match preset:
                case 'sdxl':
                    return 1.3, 1.4, 0.9, 0.2
                case 'sd2.1':
                    return 1.4, 1.6, 0.9, 0.2
                case 'sd1.5':
                    return 1.5, 1.6, 0.9, 0.2
                case 'sd1.4':
                    return 1.3, 1.4, 0.9, 0.2
                case _:
                    return 1.0, 1.0, 1.0, 1.0

        freeu_preset.select (fn=setPresetValues, inputs=[freeu_preset], outputs=[freeu_b1, freeu_b2, freeu_s1, freeu_s2])
        freeu_b1.input (fn=setPresetText, inputs=[], outputs=[freeu_preset], show_progress='hidden')
        freeu_b2.input (fn=setPresetText, inputs=[], outputs=[freeu_preset], show_progress='hidden')
        freeu_s1.input (fn=setPresetText, inputs=[], outputs=[freeu_preset], show_progress='hidden')
        freeu_s2.input (fn=setPresetText, inputs=[], outputs=[freeu_preset], show_progress='hidden')

        return freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2

    def process(self, p, *script_args, **kwargs):
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2 = script_args

        if freeu_enabled:
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

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2 = script_args

        if freeu_enabled:
            unet = p.sd_model.forge_objects.unet
            unet = patch_freeu_v2(unet, freeu_b1, freeu_b2, freeu_s1, freeu_s2)
            p.sd_model.forge_objects.unet = unet

        return
