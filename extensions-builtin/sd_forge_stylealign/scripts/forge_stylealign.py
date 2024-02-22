import torch
import gradio as gr

from modules import scripts
import ldm_patched.ldm.modules.attention as attention


def sdp(q, k, v, transformer_options):
    return attention.optimized_attention(q, k, v, heads=transformer_options["n_heads"], mask=None)


class StyleAlignForForge(scripts.Script):
    sorting_priority = 17

    def title(self):
        return "StyleAlign Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            shared_attention = gr.Checkbox(label='Share attention in batch', value=False)

        return [shared_attention]

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        shared_attention = script_args[0]

        if not shared_attention:
            return

        unet = p.sd_model.forge_objects.unet.clone()

        def join(x):
            b, f, c = x.shape
            return x.reshape(1, b * f, c)

        def aligned_attention(q, k, v, transformer_options):
            b, f, c = q.shape
            o = sdp(join(q), join(k), join(v), transformer_options)
            b2, f2, c2 = o.shape
            o = o.reshape(b, b2 * f2 // b, c2)
            return o

        def attn1_proc(q, k, v, transformer_options):
            cond_indices = transformer_options['cond_indices']
            uncond_indices = transformer_options['uncond_indices']
            cond_or_uncond = transformer_options['cond_or_uncond']
            results = []

            for cx in cond_or_uncond:
                if cx == 0:
                    indices = cond_indices
                else:
                    indices = uncond_indices

                if len(indices) > 0:
                    bq, bk, bv = q[indices], k[indices], v[indices]
                    bo = aligned_attention(bq, bk, bv, transformer_options)
                    results.append(bo)

            results = torch.cat(results, dim=0)
            return results

        unet.set_model_replace_all(attn1_proc, 'attn1')

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            stylealign_enabled=shared_attention,
        ))

        return
