import torch

from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.forge_util import numpy_to_pytorch, resize_image_with_pad
from modules_forge.shared import add_supported_preprocessor


class PreprocessorTile(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'tile_resample'
        self.tags = ['Tile']
        self.model_filename_filters = ['tile']
        self.slider_resolution = PreprocessorParameter(visible=False)


class PreprocessorTileColorFix(PreprocessorTile):
    def __init__(self):
        super().__init__()
        self.name = 'tile_colorfix'
        self.slider_1 = PreprocessorParameter(label='Variation', value=8.0, minimum=3.0, maximum=32.0, step=1.0, visible=True)


class PreprocessorTileColorFixSharp(PreprocessorTileColorFix):
    def __init__(self):
        super().__init__()
        self.name = 'tile_colorfix+sharp'
        self.slider_2 = PreprocessorParameter(label='Sharpness', value=1.0, minimum=0.0, maximum=2.0, step=0.01, visible=True)


add_supported_preprocessor(PreprocessorTile())

add_supported_preprocessor(PreprocessorTileColorFix())

add_supported_preprocessor(PreprocessorTileColorFixSharp())
