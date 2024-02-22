import gradio as gr

from modules import scripts
from ldm_patched.contrib.external_hypertile import HyperTile


opHyperTile = HyperTile()


class HyperTileForForge(scripts.Script):
    sorting_priority = 13

    def title(self):
        return "HyperTile Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            tile_size = gr.Slider(label='Tile Size', minimum=1, maximum=2048, step=1, value=256)
            swap_size = gr.Slider(label='Swap Size', minimum=1, maximum=128, step=1, value=2)
            max_depth = gr.Slider(label='Max Depth', minimum=0, maximum=10, step=1, value=0)
            scale_depth = gr.Checkbox(label='Scale Depth', value=False)

        return enabled, tile_size, swap_size, max_depth, scale_depth

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, tile_size, swap_size, max_depth, scale_depth = script_args
        tile_size, swap_size, max_depth = int(tile_size), int(swap_size), int(max_depth)

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = opHyperTile.patch(unet, tile_size, swap_size, max_depth, scale_depth)[0]

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            HyperTile_enabled=enabled,
            HyperTile_tile_size=tile_size,
            HyperTile_swap_size=swap_size,
            HyperTile_max_depth=max_depth,
            HyperTile_scale_depth=scale_depth,
        ))

        return
