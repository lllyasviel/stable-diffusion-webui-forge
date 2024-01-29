from modules_forge.shared import Preprocessor, PreprocessorParameter, preprocessor_dir, add_preprocessor

# This is a python script to convert all old preprocessors to new format.
# However, the old preprocessors are not very memory effective
# and eventually we should move all old preprocessors to new format manually
# see also the forge_preprocessor_normalbae/scripts/preprocessor_normalbae for
# how to make better implementation of preprocessors.
# No newer preprocessors should be written in this legacy way.

from legacy_preprocessors.preprocessor_meta import cn_preprocessor_modules, cn_preprocessor_unloadable, ui_preprocessor_keys, reverse_preprocessor_aliases, preprocessor_aliases
from legacy_preprocessors.preprocessor import model_free_preprocessors, no_control_mode_preprocessors, preprocessor_sliders_config, preprocessor_filters, preprocessor_filters_aliases


def special_get(d, k, default=None):
    k1 = k
    k2 = preprocessor_aliases.get(k, k)
    k3 = reverse_preprocessor_aliases.get(k, k)

    for pk in [k1, k2, k3]:
        if pk in d:
            return d[pk]

    return default


def special_judge_in(d, k):
    k1 = k
    k2 = preprocessor_aliases.get(k, k)
    k3 = reverse_preprocessor_aliases.get(k, k)

    for pk in [k1, k2, k3]:
        if pk in d:
            return True

    return False


legacy_preprocessors = {}

for name in ui_preprocessor_keys:
    call_function = special_get(cn_preprocessor_modules, name, None)
    assert call_function is not None
    unload_function = special_get(cn_preprocessor_unloadable, name, None)

    model_free = special_judge_in(model_free_preprocessors, name)
    no_control_mode = special_judge_in(no_control_mode_preprocessors, name)
    slider_config = special_get(preprocessor_sliders_config, name, [])

    resolution = slider_config[0] if len(slider_config) > 0 else None
    slider_1 = slider_config[1] if len(slider_config) > 1 else None
    slider_2 = slider_config[2] if len(slider_config) > 2 else None
    slider_3 = slider_config[3] if len(slider_config) > 3 else None

    legacy_preprocessors[name] = dict(
        name=name,
        call_function=call_function,
        unload_function=unload_function,
        model_free=model_free,
        no_control_mode=no_control_mode,
        resolution=resolution,
        slider_1=slider_1,
        slider_2=slider_2,
        slider_3=slider_3,
        priority=0,
        tag=None
    )


for tag, best in preprocessor_filters.items():
    bp = special_get(legacy_preprocessors, best, None)
    if bp is not None:
        bp['priority'] = 100

for tag, best in preprocessor_filters.items():
    marks = [tag.lower()] + preprocessor_filters_aliases.get(tag.lower(), [])
    for k, p in legacy_preprocessors.items():
        if any(x.lower() in k.lower() for x in marks):
            p['tag'] = tag

a = 0
