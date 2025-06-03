import torch
import gradio as gr

from modules import scripts
from backend import attention
from modules.ui_components import InputAccordion

def sdp(q, k, v, transformer_options):
    return attention.attention_function(q, k, v, heads=transformer_options["n_heads"], mask=None)


class StyleAlignForForge(scripts.Script):
    sorting_priority = 17

    def title(self):
        return "StyleAlign Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        elem = 'stylealign_'
        with InputAccordion(False, label=self.title()+' - Share attention in batch',elem_id=elem+'enabled') as shared_attention:
            strength = gr.Slider(label='Strength', minimum=0.0, maximum=1.0, value=1.0,elem_id=elem+'strength')

        return [shared_attention, strength]

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        shared_attention, strength = script_args

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

                    if strength < 0.01:
                        # At strength = 0, use original.
                        original_attention = sdp(bq, bk, bv, transformer_options)
                        results.append(original_attention)

                    elif strength > 0.99:
                        # At strength 1, use aligned.
                        aligned_attention_result = aligned_attention(bq, bk, bv, transformer_options)
                        results.append(aligned_attention_result)

                    else:
                        # In between, blend original and aligned attention based on strength.
                        original_attention = sdp(bq, bk, bv, transformer_options)
                        aligned_attention_result = aligned_attention(bq, bk, bv, transformer_options)
                        blended_attention = (1.0 - strength) * original_attention + strength * aligned_attention_result
                        results.append(blended_attention)


            results = torch.cat(results, dim=0)
            return results

        unet.set_model_replace_all(attn1_proc, 'attn1')

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            stylealign_enabled=shared_attention,
            stylealign_strength=strength,
        ))

        return
