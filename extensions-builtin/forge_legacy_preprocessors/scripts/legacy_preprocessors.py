from modules_forge.shared import Preprocessor, PreprocessorParameter, preprocessor_dir, add_preprocessor

# This is a python script to convert all old preprocessors to new format.
# However, the old preprocessors are not very memory effective
# and eventually we should move all old preprocessors to new format manually
# see also the forge_preprocessor_normalbae/scripts/preprocessor_normalbae for
# how to make better implementation of preprocessors.
# No newer preprocessors should be written in this legacy way.

from legacy_preprocessors.preprocessor_compiled import legacy_preprocessors


class LegacyPreprocessor(Preprocessor):
    def __init__(self, legacy_dict):
        super().__init__()
        self.name = legacy_dict['name']
        self.call_function = legacy_dict['call_function']
        self.unload_function = legacy_dict['unload_function']
        self.managed_model = legacy_dict['managed_model']
        self.do_not_need_model = legacy_dict['model_free']
        self.show_control_mode = not legacy_dict['no_control_mode']
        self.sorting_priority = legacy_dict['priority']
        self.tags = legacy_dict['tags']

        if legacy_dict['resolution'] is None:
            self.resolution = PreprocessorParameter(visible=False)
        else:
            self.resolution = PreprocessorParameter(**legacy_dict['resolution'], visible=True)

        if legacy_dict['slider_1'] is None:
            self.slider_1 = PreprocessorParameter(visible=False)
        else:
            self.slider_1 = PreprocessorParameter(**legacy_dict['slider_1'], visible=True)

        if legacy_dict['slider_2'] is None:
            self.slider_2 = PreprocessorParameter(visible=False)
        else:
            self.slider_2 = PreprocessorParameter(**legacy_dict['slider_2'], visible=True)

        if legacy_dict['slider_3'] is None:
            self.slider_3 = PreprocessorParameter(visible=False)
        else:
            self.slider_3 = PreprocessorParameter(**legacy_dict['slider_3'], visible=True)


for k, v in legacy_preprocessors.items():
    p = LegacyPreprocessor(v)
    p.name = k
    add_preprocessor(p)
