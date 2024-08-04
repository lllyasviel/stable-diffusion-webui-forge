import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch import nn


class ModuleDict(torch.nn.Module):
    def __init__(self, module_dict):
        super(ModuleDict, self).__init__()
        for name, module in module_dict.items():
            self.add_module(name, module)


class ObjectDict:
    def __init__(self, module_dict):
        for name, module in module_dict.items():
            setattr(self, name, module)


class Dummy(nn.Module, ConfigMixin):
    config_name = 'config.json'

    @register_to_config
    def __init__(self):
        super().__init__()
