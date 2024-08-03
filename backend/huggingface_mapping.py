from backend import latent_spaces


class SupportedModel:
    unet_config = {}
    latent = latent_spaces.LatentSpace
    huggingface_mappings = []


class SD15(SupportedModel):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
    }
    latent = latent_spaces.SD15
    huggingface_mappings = [
        "runwayml/stable-diffusion-v1-5",
        "runwayml/stable-diffusion-inpainting"
    ]


class SD21(SupportedModel):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": None,
    }
    latent = latent_spaces.SD15
    huggingface_mappings = [
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-2-inpainting"
    ]


class SDXL(SupportedModel):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
    }
    latent = latent_spaces.SDXL
    huggingface_mappings = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "playgroundai/playground-v2.5-1024px-aesthetic",

    ]


class SD3(SupportedModel):
    unet_config = {}
    latent = latent_spaces.SD3
    huggingface_mappings = [
        "stabilityai/stable-diffusion-3-medium-diffusers"
    ]


class Flux(SupportedModel):
    pass
