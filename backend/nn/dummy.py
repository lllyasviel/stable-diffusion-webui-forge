import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch import nn


class Dummy(nn.Module, ConfigMixin):
    config_name = 'config.json'

    @register_to_config
    def __init__(self):
        super().__init__()
