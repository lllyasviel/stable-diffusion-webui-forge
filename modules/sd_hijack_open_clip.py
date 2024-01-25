import torch

from modules import sd_hijack_clip
from modules.shared import opts


class FrozenOpenCLIPEmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)


class FrozenOpenCLIPEmbedder2WithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        a = 0
