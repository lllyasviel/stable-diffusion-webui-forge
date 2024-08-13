import packages_3rdparty.webui_lora_collection.lora as lora_utils_webui
import packages_3rdparty.comfyui_lora_collection.lora as lora_utils_comfyui


class ForgeLoraCollection:
    # TODO
    pass


lora_utils_forge = ForgeLoraCollection()

lora_collection_priority = [lora_utils_forge, lora_utils_webui, lora_utils_comfyui]


def get_function(function_name: str):
    for lora_collection in lora_collection_priority:
        if hasattr(lora_collection, function_name):
            return getattr(lora_collection, function_name)


def load_lora(lora, to_load):
    patch_dict, remaining_dict = get_function('load_lora')(lora, to_load)
    return patch_dict, remaining_dict


def model_lora_keys_clip(model, key_map={}):
    return get_function('model_lora_keys_clip')(model, key_map)


def model_lora_keys_unet(model, key_map={}):
    return get_function('model_lora_keys_unet')(model, key_map)
