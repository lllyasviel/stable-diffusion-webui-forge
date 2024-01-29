import os.path
import stat
import functools
from collections import OrderedDict

from modules import shared, scripts, sd_models
from scripts.utils import ndarray_lru_cache
from scripts.logging import logger
from scripts.enums import StableDiffusionVersion

from typing import Dict, Callable, Optional, Tuple, List


CN_MODEL_EXTS = [".pt", ".pth", ".ckpt", ".safetensors", ".bin"]


def cache_preprocessors(preprocessor_modules: Dict[str, Callable]) -> Dict[str, Callable]:
    """ We want to share the preprocessor results in a single big cache, instead of a small
     cache for each preprocessor function. """
    CACHE_SIZE = getattr(shared.cmd_opts, "controlnet_preprocessor_cache_size", 0)

    # Set CACHE_SIZE = 0 will completely remove the caching layer. This can be
    # helpful when debugging preprocessor code.
    if CACHE_SIZE == 0:
        return preprocessor_modules

    logger.debug(f'Create LRU cache (max_size={CACHE_SIZE}) for preprocessor results.')

    @ndarray_lru_cache(max_size=CACHE_SIZE)
    def unified_preprocessor(preprocessor_name: str, *args, **kwargs):
        logger.debug(f'Calling preprocessor {preprocessor_name} outside of cache.')
        return preprocessor_modules[preprocessor_name](*args, **kwargs)

    # TODO: Introduce a seed parameter for shuffle preprocessor?
    uncacheable_preprocessors = ['shuffle']

    return {
        k: (
            v if k in uncacheable_preprocessors
            else functools.partial(unified_preprocessor, k)
        )
        for k, v
        in preprocessor_modules.items()
    }


default_detectedmap_dir = os.path.join("detected_maps")
script_dir = scripts.basedir()


def traverse_all_files(curr_path, model_list):
    f_list = [
        (os.path.join(curr_path, entry.name), entry.stat())
        for entry in os.scandir(curr_path)
        if os.path.isdir(curr_path)
    ]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in CN_MODEL_EXTS:
            model_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            model_list = traverse_all_files(fname, model_list)
    return model_list


def get_all_models(sort_by, filter_by, path):
    res = OrderedDict()
    fileinfos = traverse_all_files(path, [])
    filter_by = filter_by.strip(" ")
    if len(filter_by) != 0:
        fileinfos = [x for x in fileinfos if filter_by.lower()
                     in os.path.basename(x[0]).lower()]
    if sort_by == "name":
        fileinfos = sorted(fileinfos, key=lambda x: os.path.basename(x[0]))
    elif sort_by == "date":
        fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
    elif sort_by == "path name":
        fileinfos = sorted(fileinfos)

    for finfo in fileinfos:
        filename = finfo[0]
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name + f" [{sd_models.model_hash(filename)}]"] = filename

    return res


def update_cn_models():
    cn_models.clear()
    ext_dirs = (shared.opts.data.get("control_net_models_path", None), getattr(shared.cmd_opts, 'controlnet_dir', None))
    extra_lora_paths = (extra_lora_path for extra_lora_path in ext_dirs
                if extra_lora_path is not None and os.path.exists(extra_lora_path))
    paths = [cn_models_dir, cn_models_dir_old, *extra_lora_paths]

    for path in paths:
        sort_by = shared.opts.data.get(
            "control_net_models_sort_models_by", "name")
        filter_by = shared.opts.data.get("control_net_models_name_filter", "")
        found = get_all_models(sort_by, filter_by, path)
        cn_models.update({**found, **cn_models})

    # insert "None" at the beginning of `cn_models` in-place
    cn_models_copy = OrderedDict(cn_models)
    cn_models.clear()
    cn_models.update({**{"None": None}, **cn_models_copy})

    cn_models_names.clear()
    for name_and_hash, filename in cn_models.items():
        if filename is None:
            continue
        name = os.path.splitext(os.path.basename(filename))[0].lower()
        cn_models_names[name] = name_and_hash


def get_sd_version() -> StableDiffusionVersion:
    if shared.sd_model.is_sdxl:
        return StableDiffusionVersion.SDXL
    elif shared.sd_model.is_sd2:
        return StableDiffusionVersion.SD2x
    elif shared.sd_model.is_sd1:
        return StableDiffusionVersion.SD1x
    else:
        return StableDiffusionVersion.UNKNOWN


def select_control_type(
    control_type: str,
    sd_version: StableDiffusionVersion = StableDiffusionVersion.UNKNOWN,
    cn_models: Dict = cn_models, # Override or testing
) -> Tuple[List[str], List[str], str, str]:
    default_option = processor.preprocessor_filters[control_type]
    pattern = control_type.lower()
    preprocessor_list = ui_preprocessor_keys
    all_models = list(cn_models.keys())

    if pattern == "all":
        return [
            preprocessor_list,
            all_models,
            'none', #default option
            "None"  #default model
        ]
    filtered_preprocessor_list = [
        x
        for x in preprocessor_list
        if ((
            pattern in x.lower() or
            any(a in x.lower() for a in processor.preprocessor_filters_aliases.get(pattern, [])) or
            x.lower() == "none"
        ) and (
            sd_version.is_compatible_with(StableDiffusionVersion.detect_from_model_name(x))
        ))
    ]
    if pattern in ["canny", "lineart", "scribble/sketch", "mlsd"]:
        filtered_preprocessor_list += [
            x for x in preprocessor_list if "invert" in x.lower()
        ]
    filtered_model_list = [
        model for model in all_models
        if model.lower() == "none" or
        ((
            pattern in model.lower() or
            any(a in model.lower() for a in processor.preprocessor_filters_aliases.get(pattern, []))
        ) and (
            sd_version.is_compatible_with(StableDiffusionVersion.detect_from_model_name(model))
        ))
    ]
    assert len(filtered_model_list) > 0, "'None' model should always be available."
    if default_option not in filtered_preprocessor_list:
        default_option = filtered_preprocessor_list[0]
    if len(filtered_model_list) == 1:
        default_model = "None"
    else:
        default_model = filtered_model_list[1]
        for x in filtered_model_list:
            if "11" in x.split("[")[0]:
                default_model = x
                break

    return (
        filtered_preprocessor_list,
        filtered_model_list,
        default_option,
        default_model
    )


ip_adapter_pairing_model = {
    "ip-adapter_clip_sdxl": lambda model: "faceid" not in model and "vit" not in model,
    "ip-adapter_clip_sdxl_plus_vith": lambda model: "faceid" not in model and "vit" in model,
    "ip-adapter_clip_sd15": lambda model: "faceid" not in model,
    "ip-adapter_face_id": lambda model: "faceid" in model and "plus" not in model,
    "ip-adapter_face_id_plus": lambda model: "faceid" in model and "plus" in model,
}

ip_adapter_pairing_logic_text = """
{
    "ip-adapter_clip_sdxl": lambda model: "faceid" not in model and "vit" not in model,
    "ip-adapter_clip_sdxl_plus_vith": lambda model: "faceid" not in model and "vit" in model,
    "ip-adapter_clip_sd15": lambda model: "faceid" not in model,
    "ip-adapter_face_id": lambda model: "faceid" in model and "plus" not in model,
    "ip-adapter_face_id_plus": lambda model: "faceid" in model and "plus" in model,
}
"""
