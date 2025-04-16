import re
import threading
import hashlib

from PIL import Image, ImageFilter, ImageOps
import numpy as np

from modules import scripts_postprocessing, shared
import gradio as gr

from modules.ui_components import FormRow, ToolButton, InputAccordion
from modules.ui import switch_values_symbol

# Simple in-memory cache with thread safety
upscale_cache = {}
cache_lock = threading.Lock()

def limit_size_by_one_dimension(w, h, limit):
    """Resize w,h so that the larger dimension == limit, preserving aspect ratio."""
    if h > w and h > limit:
        w = limit * w // h
        h = limit
    elif w >= h and w > limit:
        h = limit * h // w
        w = limit
    return int(w), int(h)

class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Upscale"
    order = 1000

    def ui(self):
        selected_tab = gr.Number(value=0, visible=False)
        with InputAccordion(True, label="Upscale & Post‑Process", elem_id="extras_upscale") as upscale_enabled:
            # Upscale settings
            with FormRow():
                extras_upscaler_1 = gr.Dropdown(
                    label='Upscaler 1', choices=[x.name for x in shared.sd_upscalers],
                    value=shared.sd_upscalers[0].name, elem_id="extras_upscaler_1"
                )
                extras_upscaler_2 = gr.Dropdown(
                    label='Upscaler 2', choices=[x.name for x in shared.sd_upscalers],
                    value=shared.sd_upscalers[0].name, elem_id="extras_upscaler_2"
                )
                extras_upscaler_2_visibility = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, label="Upscaler 2 blend",
                    value=0.0, elem_id="extras_upscaler_2_visibility"
                )

            with FormRow():
                with gr.Tabs(elem_id="extras_resize_mode"):
                    with gr.TabItem('Scale by') as tab_scale_by:
                        upscaling_resize = gr.Slider(
                            minimum=1.0, maximum=8.0, step=0.05, label="Scale factor",
                            value=2.0, elem_id="extras_upscaling_resize"
                        )
                        max_side_length = gr.Number(
                            label="Max side length", value=0,
                            elem_id="extras_upscale_max_side_length",
                            tooltip="Limit largest side after scaling (0=none)"
                        )
                    with gr.TabItem('Scale to') as tab_scale_to:
                        upscaling_resize_w = gr.Slider(
                            minimum=64, maximum=8192, step=8, label="Width",
                            value=512, elem_id="extras_upscaling_resize_w"
                        )
                        upscaling_resize_h = gr.Slider(
                            minimum=64, maximum=8192, step=8, label="Height",
                            value=512, elem_id="extras_upscaling_resize_h"
                        )
                        upscaling_crop = gr.Checkbox(
                            label='Crop to fit', value=True, elem_id="extras_upscaling_crop"
                        )
                        ToolButton(
                            value=switch_values_symbol, elem_id="upscaling_res_switch_btn",
                            tooltip="Swap width/height"
                        ).click(
                            lambda w, h: (h, w),
                            inputs=[upscaling_resize_w, upscaling_resize_h],
                            outputs=[upscaling_resize_w, upscaling_resize_h]
                        )

            # New post-processing options
            with FormRow():
                denoise_method = gr.Dropdown(
                    label="Denoise method",
                    choices=["none", "median", "gaussian"],
                    value="none", elem_id="extras_denoise_method"
                )
                denoise_strength = gr.Slider(
                    minimum=1, maximum=10, step=1, label="Denoise strength",
                    value=3, elem_id="extras_denoise_strength"
                )
                sharpen_strength = gr.Slider(
                    minimum=0.0, maximum=5.0, step=0.1, label="Sharpen strength",
                    value=1.0, elem_id="extras_sharpen_strength"
                )
                hist_equalize = gr.Checkbox(
                    label="Histogram equalization", value=False,
                    elem_id="extras_hist_equalize"
                )

            # Cache controls
            clear_cache_btn = ToolButton("🗑️ Clear cache", elem_id="extras_clear_cache")
            clear_cache_btn.click(lambda: (upscale_cache.clear(), gr.update()), outputs=[])

        # wire tab selector
        tab_scale_by.select(fn=lambda:0, inputs=[], outputs=[selected_tab])
        tab_scale_to.select(fn=lambda:1, inputs=[], outputs=[selected_tab])

        return {
            "upscale_enabled": upscale_enabled,
            "upscale_mode": selected_tab,
            "upscale_by": upscaling_resize,
            "max_side_length": max_side_length,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
            "denoise_method": denoise_method,
            "denoise_strength": denoise_strength,
            "sharpen_strength": sharpen_strength,
            "hist_equalize": hist_equalize,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # New Utility Functions
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def denoise(image: Image.Image, method: str, strength: int) -> Image.Image:
        if method == "median":
            return image.filter(ImageFilter.MedianFilter(size=strength))
        elif method == "gaussian":
            return image.filter(ImageFilter.GaussianBlur(radius=strength/2))
        return image

    @staticmethod
    def sharpen(image: Image.Image, strength: float) -> Image.Image:
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=int(strength*100), threshold=3))

    @staticmethod
    def equalize_histogram(image: Image.Image) -> Image.Image:
        if image.mode != "L":
            # convert each channel separately
            channels = image.split()
            eq = [ImageOps.equalize(c) for c in channels]
            return Image.merge(image.mode, eq)
        return ImageOps.equalize(image)

    @staticmethod
    def tile_upscale(image: Image.Image, tile_size: int = 512, overlap: int = 32):
        """Upscale large images by tiling to avoid OOM."""
        w, h = image.size
        out = Image.new(image.mode, (w, h))
        for x in range(0, w, tile_size - overlap):
            for y in range(0, h, tile_size - overlap):
                box = (x, y, min(x+tile_size, w), min(y+tile_size, h))
                tile = image.crop(box)
                up = shared.sd_upscalers[0].scaler.upscale(tile, 2, shared.sd_upscalers[0].data_path)
                out.paste(up.resize(tile.size), box)
        return out

    @staticmethod
    def measure_similarity(img1: Image.Image, img2: Image.Image) -> float:
        """Compute simple MSE similarity."""
        a = np.array(img1).astype(np.float32)
        b = np.array(img2).astype(np.float32)
        return float(np.mean((a - b) ** 2))

    @staticmethod
    def suggest_upscale_factor(image: Image.Image, target_max_dim: int = 1024) -> float:
        w, h = image.size
        return min(target_max_dim / w, target_max_dim / h)

    # ──────────────────────────────────────────────────────────────────────────
    # Core Upscale Method (extended)
    # ──────────────────────────────────────────────────────────────────────────
    def upscale(self, image, info, upscaler, upscale_mode, upscale_by,
                max_side_length, upscale_to_width, upscale_to_height,
                upscale_crop, denoise_method, denoise_strength,
                sharpen_strength, hist_equalize):
        # Determine final scale
        if upscale_mode == 1:
            upscale_by = max(upscale_to_width / image.width,
                             upscale_to_height / image.height)
            info["Upscale to"] = f"{upscale_to_width}x{upscale_to_height}"
        else:
            info["Upscale by"] = upscale_by
            if max_side_length and max(*image.size) * upscale_by > max_side_length:
                upscale_mode = 1
                upscale_crop = False
                w2, h2 = limit_size_by_one_dimension(
                    image.width * upscale_by,
                    image.height * upscale_by,
                    max_side_length
                )
                upscale_to_width, upscale_to_height = w2, h2
                upscale_by = max(w2 / image.width, h2 / image.height)
                info["Max side length"] = max_side_length

        # cache key
        key = hashlib.sha256(
            np.array(image).tobytes() +
            upscaler.name.encode() +
            f"{upscale_mode}-{upscale_by}-{upscale_crop}".encode()
        ).hexdigest()

        with cache_lock:
            cached = upscale_cache.get(key)
        if cached:
            out = cached
        else:
            out = upscaler.scaler.upscale(image, upscale_by, upscaler.data_path)
            with cache_lock:
                upscale_cache[key] = out
                # prune if too big
                if len(upscale_cache) > shared.opts.upscaling_max_images_in_cache:
                    upscale_cache.pop(next(iter(upscale_cache)))

        # crop if needed
        if upscale_mode == 1 and upscale_crop:
            crop_box = (
                (out.width - upscale_to_width)//2,
                (out.height - upscale_to_height)//2,
                (out.width + upscale_to_width)//2,
                (out.height + upscale_to_height)//2
            )
            out = out.crop(crop_box)
            info["Crop to"] = f"{out.width}x{out.height}"

        # Post-processing
        if denoise_method != "none":
            out = self.denoise(out, denoise_method, denoise_strength)
            info["Denoised"] = f"{denoise_method} (s={denoise_strength})"
        if hist_equalize:
            out = self.equalize_histogram(out)
            info["Histogram equalized"] = True
        if sharpen_strength > 0:
            out = self.sharpen(out, sharpen_strength)
            info["Sharpened"] = f"{sharpen_strength}"

        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Hooks
    # ──────────────────────────────────────────────────────────────────────────
    def process_firstpass(self, pp, upscale_enabled=True, **kwargs):
        if not upscale_enabled:
            return
        # set target dimensions for SD
        mode = kwargs.get("upscale_mode", 1)
        if mode == 1:
            pp.shared.target_width = kwargs["upscale_to_width"]
            pp.shared.target_height = kwargs["upscale_to_height"]
        else:
            w = int(pp.image.width * kwargs["upscale_by"])
            h = int(pp.image.height * kwargs["upscale_by"])
            pp.shared.target_width, pp.shared.target_height = limit_size_by_one_dimension(w, h, kwargs["max_side_length"])

    def process(self, pp, upscale_enabled=True, **kwargs):
        if upscale_enabled:
            up1 = next(x for x in shared.sd_upscalers if x.name == kwargs["upscaler_1_name"])
            img = self.upscale(pp.image, pp.info, up1, **kwargs)
            # optional second pass
            name2 = kwargs["upscaler_2_name"]
            v2 = kwargs["upscaler_2_visibility"]
            if name2 and v2 > 0:
                up2 = next(x for x in shared.sd_upscalers if x.name == name2)
                img2 = self.upscale(pp.image, pp.info, up2, **kwargs)
                if img2.mode != img.mode:
                    img2 = img2.convert(img.mode)
                img = Image.blend(img, img2, v2)
                pp.info["Blended with"] = name2
            pp.image = img

    def image_changed(self):
        with cache_lock:
            upscale_cache.clear()
