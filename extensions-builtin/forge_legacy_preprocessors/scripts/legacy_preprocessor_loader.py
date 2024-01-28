from modules_forge.shared import Preprocessor, PreprocessorParameter, preprocessor_dir, load_file_from_url, add_preprocessor

# This is a python script to convert all old preprocessors to new format.
# However, the old preprocessors are not very memory effective
# and eventually we should move all old preprocessors to new format manually
# see also the forge_preprocessor_normalbae/scripts/preprocessor_normalbae for
# how to make better implementation of preprocessors.
# No newer preprocessors should be written in this legacy way.

from legacy_preprocessors.preprocessor_meta import cn_preprocessor_modules, cn_preprocessor_unloadable, ui_preprocessor_keys, reverse_preprocessor_aliases
from legacy_preprocessors.preprocessor import model_free_preprocessors, no_control_mode_preprocessors, flag_preprocessor_resolution, preprocessor_sliders_config, preprocessor_filters, preprocessor_filters_aliases


class LegacyPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        self.legacy_call_function = None
        self.legacy_unload_function = None
        return


legacy_preprocessors = {}

for k in ui_preprocessor_keys:
    p = LegacyPreprocessor()
    p.name = k
    real_key = reverse_preprocessor_aliases.get(k, k)
    if real_key not in cn_preprocessor_modules:
        print(f'bad key {real_key}')
        continue
    p.legacy_call_function = cn_preprocessor_modules[real_key]
    p.legacy_unload_function = cn_preprocessor_unloadable.get(real_key, lambda: None)
    legacy_preprocessors[k] = p


a = 0
