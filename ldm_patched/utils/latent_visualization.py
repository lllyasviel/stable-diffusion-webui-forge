# Taken from https://github.com/comfyanonymous/ComfyUI
# This file is only for reference, and not used in the backend or runtime.

import torch
from PIL import Image
import struct
import numpy as np
from ldm_patched.modules.args_parser import args, LatentPreviewMethod
from ldm_patched.taesd.taesd import TAESD
import ldm_patched.utils.path_utils
import ldm_patched.modules.utils

MAX_PREVIEW_RESOLUTION = 512

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)

class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1])[0].detach()
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
        x_sample = x_sample.astype(np.uint8)

        preview_image = Image.fromarray(x_sample)
        return preview_image


class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu")

    def decode_latent_to_preview(self, x0):
        latent_image = x0[0].permute(1, 2, 0).cpu() @ self.latent_rgb_factors

        latents_ubyte = (((latent_image + 1) / 2)
                            .clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            .byte()).cpu()

        return Image.fromarray(latents_ubyte.numpy())


def get_previewer(device, latent_format):
    previewer = None
    method = args.preview_option
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer methods
        taesd_decoder_path = None
        if latent_format.taesd_decoder_name is not None:
            taesd_decoder_path = next(
                (fn for fn in ldm_patched.utils.path_utils.get_filename_list("vae_approx")
                    if fn.startswith(latent_format.taesd_decoder_name)),
                ""
            )
            taesd_decoder_path = ldm_patched.utils.path_utils.get_full_path("vae_approx", taesd_decoder_path)

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB
            if taesd_decoder_path:
                method = LatentPreviewMethod.TAESD

        if method == LatentPreviewMethod.TAESD:
            if taesd_decoder_path:
                taesd = TAESD(None, taesd_decoder_path).to(device)
                previewer = TAESDPreviewerImpl(taesd)
            else:
                print("Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(latent_format.taesd_decoder_name))

        if previewer is None:
            if latent_format.latent_rgb_factors is not None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors)
    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer(model.load_device, model.model.latent_format)

    pbar = ldm_patched.modules.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

