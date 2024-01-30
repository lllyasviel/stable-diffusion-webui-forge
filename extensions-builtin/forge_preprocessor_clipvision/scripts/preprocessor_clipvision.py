from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import preprocessor_dir, add_supported_preprocessor
from modules.modelloader import load_file_from_url


class PreprocessorClipVision(Preprocessor):
    def __init__(self):
        super().__init__()


add_supported_preprocessor(PreprocessorClipVision())
