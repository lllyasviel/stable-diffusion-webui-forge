from modules_forge.supported_preprocessor import Preprocessor
from modules_forge.shared import preprocessor_dir, add_supported_preprocessor
from modules.modelloader import load_file_from_url
from modules_forge.forge_util import numpy_to_pytorch

import ldm_patched.modules.clip_vision


class PreprocessorClipVision(Preprocessor):
    def __init__(self, name, url, filename):
        super().__init__()
        self.name = name
        self.url = url
        self.filename = filename
        self.tags = ['IP-Adapter']
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = False
        self.show_control_mode = False
        self.sorting_priority = 1
        self.clipvision = None

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        if self.clipvision is None:
            ckpt_path = load_file_from_url(
                url=self.url,
                model_dir=preprocessor_dir,
                file_name=self.filename
            )
            self.clipvision = ldm_patched.modules.clip_vision.load(ckpt_path)

        input_image = numpy_to_pytorch(input_image)

        return self.clipvision.encode_image(input_image)


add_supported_preprocessor(PreprocessorClipVision(
    name='CLIP-ViT-H',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-H-14.safetensors'
))

add_supported_preprocessor(PreprocessorClipVision(
    name='CLIP-ViT-bigG',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors'
))

add_supported_preprocessor(PreprocessorClipVision(
    name='CLIP-ViT-L',
    url='https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin',
    filename='CLIP-ViT-bigG.safetensors'
))
