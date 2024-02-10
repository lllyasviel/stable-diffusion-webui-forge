import os.path
import stat
from collections import OrderedDict

from modules import shared, sd_models
from lib_controlnet.enums import StableDiffusionVersion
from modules_forge.shared import controlnet_dir, supported_preprocessors


CN_MODEL_EXTS = [".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".patch"]


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


controlnet_filename_dict = {'None': 'model.safetensors'}
controlnet_names = ['None']


def get_preprocessor(name):
    return supported_preprocessors.get(name, None)


def get_sorted_preprocessors():
    preprocessors = [p for k, p in supported_preprocessors.items() if k != 'None']
    preprocessors = sorted(preprocessors, key=lambda x: str(x.sorting_priority).zfill(8) + x.name)[::-1]
    results = OrderedDict()
    results['None'] = supported_preprocessors['None']
    for p in preprocessors:
        results[p.name] = p
    return results


def get_all_controlnet_names():
    return controlnet_names


def get_controlnet_filename(controlnet_name):
    return controlnet_filename_dict[controlnet_name]


def get_all_preprocessor_names():
    return list(get_sorted_preprocessors().keys())


def get_all_preprocessor_tags():
    tags = []
    for k, p in supported_preprocessors.items():
        tags += p.tags
    tags = list(set(tags))
    tags = sorted(tags)
    return ['All'] + tags


def get_filtered_preprocessors(tag):
    if tag == 'All':
        return supported_preprocessors
    return {k: v for k, v in get_sorted_preprocessors().items() if tag in v.tags or k == 'None'}


def get_filtered_preprocessor_names(tag):
    return list(get_filtered_preprocessors(tag).keys())


def get_filtered_controlnet_names(tag):
    filtered_preprocessors = get_filtered_preprocessors(tag)
    model_filename_filters = []
    for p in filtered_preprocessors.values():
        model_filename_filters += p.model_filename_filters
    return [
        x for x in controlnet_names
        if x == 'None' or (
            any(f.lower() in x.lower() for f in model_filename_filters) and
            get_sd_version().is_compatible_with(StableDiffusionVersion.detect_from_model_name(x))
        )
    ]


def update_controlnet_filenames():
    global controlnet_filename_dict, controlnet_names

    controlnet_filename_dict = {'None': 'model.safetensors'}
    controlnet_names = ['None']

    ext_dirs = (shared.opts.data.get("control_net_models_path", None), getattr(shared.cmd_opts, 'controlnet_dir', None))
    extra_lora_paths = (extra_lora_path for extra_lora_path in ext_dirs
                        if extra_lora_path is not None and os.path.exists(extra_lora_path))
    paths = [controlnet_dir, *extra_lora_paths]

    for path in paths:
        sort_by = shared.opts.data.get("control_net_models_sort_models_by", "name")
        filter_by = shared.opts.data.get("control_net_models_name_filter", "")
        found = get_all_models(sort_by, filter_by, path)
        controlnet_filename_dict.update(found)

    controlnet_names = list(controlnet_filename_dict.keys())
    return


def get_sd_version() -> StableDiffusionVersion:
    if not shared.sd_model:
        return StableDiffusionVersion.UNKNOWN
    if shared.sd_model.is_sdxl:
        return StableDiffusionVersion.SDXL
    elif shared.sd_model.is_sd2:
        return StableDiffusionVersion.SD2x
    elif shared.sd_model.is_sd1:
        return StableDiffusionVersion.SD1x
    else:
        return StableDiffusionVersion.UNKNOWN
