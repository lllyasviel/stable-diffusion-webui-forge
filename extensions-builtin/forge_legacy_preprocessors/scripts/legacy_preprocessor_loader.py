from modules_forge.shared import Preprocessor, PreprocessorParameter, preprocessor_dir, load_file_from_url, add_preprocessor

# This is a python script to convert all old preprocessors to new format.
# However, the old preprocessors are not very memory effective
# and eventually we should move all old preprocessors to new format manually
# see also the forge_preprocessor_normalbae/scripts/preprocessor_normalbae for
# how to make better implementation of preprocessors.
# No newer preprocessors should be written in this legacy way.

from legacy_preprocessors.preprocessor_meta import cn_preprocessor_modules, cn_preprocessor_unloadable


class LegacyPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        legacy_call_function = None
        legacy_unload_function = None
        return


legacy_preprocessors = {}

for k, v in cn_preprocessor_modules.items():
    p = LegacyPreprocessor()
    p.name = k
    p.legacy_call_function = v
    legacy_preprocessors[k] = p


a = 0
