import os

import torch
from torch import nn
from modules import devices, paths, shared

sd_vae_approx_models = {}


class VAEApprox(nn.Module):
    def __init__(self, latent_channels=4):
        super(VAEApprox, self).__init__()
        self.conv1 = nn.Conv2d(latent_channels, 8, (7, 7))
        self.conv2 = nn.Conv2d(8, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 32, (3, 3))
        self.conv6 = nn.Conv2d(32, 16, (3, 3))
        self.conv7 = nn.Conv2d(16, 8, (3, 3))
        self.conv8 = nn.Conv2d(8, 3, (3, 3))

    def forward(self, x):
        extra = 11
        x = nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = nn.functional.pad(x, (extra, extra, extra, extra))

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, ]:
            x = layer(x)
            x = nn.functional.leaky_relu(x, 0.1)

        return x


def download_model(model_path, model_url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        print(f'Downloading VAEApprox model to: {model_path}')
        torch.hub.download_url_to_file(model_url, model_path)


def model():
    if not shared.sd_model.is_webui_legacy_model():
        return None

    if shared.sd_model.is_sd3:
        model_name = "vaeapprox-sd3.pt"
    elif shared.sd_model.is_sdxl:
        model_name = "vaeapprox-sdxl.pt"
    else:
        model_name = "model.pt"

    loaded_model = sd_vae_approx_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths.models_path, "VAE-approx", model_name)
        if not os.path.exists(model_path):
            model_path = os.path.join(paths.script_path, "models", "VAE-approx", model_name)

        if not os.path.exists(model_path):
            model_path = os.path.join(paths.models_path, "VAE-approx", model_name)
            download_model(model_path, 'https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/' + model_name)

        loaded_model = VAEApprox(latent_channels=shared.sd_model.forge_objects.vae.latent_channels)
        loaded_model.load_state_dict(torch.load(model_path, map_location='cpu' if devices.device.type != 'cuda' else None))
        loaded_model.eval()
        loaded_model.to(devices.device, devices.dtype)
        sd_vae_approx_models[model_name] = loaded_model

    return loaded_model


def cheap_approximation(sample):
    return torch.einsum("...lxy,lr -> ...rxy", sample, torch.tensor(shared.sd_model.model_config.latent_format.latent_rgb_factors).to(sample.device))
