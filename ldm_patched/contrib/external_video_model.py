# Taken from https://github.com/comfyanonymous/ComfyUI
# This file is only for reference, and not used in the backend or runtime.


import ldm_patched.contrib.external
import torch
import ldm_patched.modules.utils
import ldm_patched.modules.sd
import ldm_patched.utils.path_utils
import ldm_patched.contrib.external_model_merging


class ImageOnlyCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (ldm_patched.utils.path_utils.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP_VISION", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders/video_models"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = ldm_patched.utils.path_utils.get_full_path("checkpoints", ckpt_name)
        out = ldm_patched.modules.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=False, output_clipvision=True, embedding_directory=ldm_patched.utils.path_utils.get_folder_paths("embeddings"))
        return (out[0], out[3], out[2])


class SVD_img2vid_Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 1024, "min": 16, "max": ldm_patched.contrib.external.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 576, "min": 16, "max": ldm_patched.contrib.external.MAX_RESOLUTION, "step": 8}),
                              "video_frames": ("INT", {"default": 14, "min": 1, "max": 4096}),
                              "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023}),
                              "fps": ("INT", {"default": 6, "min": 1, "max": 1024}),
                              "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, clip_vision, init_image, vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = ldm_patched.modules.utils.common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        t = vae.encode(encode_pixels)
        positive = [[pooled, {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level, "concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level, "concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([video_frames, 4, height // 8, width // 8])
        return (positive, negative, {"samples":latent})

class VideoLinearCFGGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "min_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "sampling/video_models"

    def patch(self, model, min_cfg):
        def linear_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]

            scale = torch.linspace(min_cfg, cond_scale, cond.shape[0], device=cond.device).reshape((cond.shape[0], 1, 1, 1))
            return uncond + scale * (cond - uncond)

        m = model.clone()
        m.set_model_sampler_cfg_function(linear_cfg)
        return (m, )

class ImageOnlyCheckpointSave(ldm_patched.contrib.external_model_merging.CheckpointSave):
    CATEGORY = "_for_testing"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip_vision": ("CLIP_VISION",),
                              "vae": ("VAE",),
                              "filename_prefix": ("STRING", {"default": "checkpoints/ldm_patched"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}

    def save(self, model, clip_vision, vae, filename_prefix, prompt=None, extra_pnginfo=None):
        ldm_patched.contrib.external_model_merging.save_checkpoint(model, clip_vision=clip_vision, vae=vae, filename_prefix=filename_prefix, output_dir=self.output_dir, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return {}

NODE_CLASS_MAPPINGS = {
    "ImageOnlyCheckpointLoader": ImageOnlyCheckpointLoader,
    "SVD_img2vid_Conditioning": SVD_img2vid_Conditioning,
    "VideoLinearCFGGuidance": VideoLinearCFGGuidance,
    "ImageOnlyCheckpointSave": ImageOnlyCheckpointSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageOnlyCheckpointLoader": "Image Only Checkpoint Loader (img2vid model)",
}
