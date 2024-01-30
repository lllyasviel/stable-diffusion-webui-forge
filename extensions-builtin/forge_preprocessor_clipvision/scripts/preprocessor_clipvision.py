from modules_forge.supported_preprocessor import PreprocessorClipVision
from modules_forge.shared import add_supported_preprocessor
from modules_forge.forge_util import numpy_to_pytorch


class PreprocessorClipVisionForIPAdapter(PreprocessorClipVision):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.tags = ['IP-Adapter']
        self.model_filename_filters = ['IP-Adapter', 'IP_Adapter']

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        clipvision = self.load_clipvision()
        return clipvision, numpy_to_pytorch(input_image)


add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-H (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-H-14.safetensors'
))

add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-bigG (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors'
))

add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-L (IPAdapter)',
    url='https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin',
    filename='CLIP-ViT-bigG.safetensors'
))
