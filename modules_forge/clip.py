from backend import memory_management
from modules import sd_models


def move_clip_to_gpu():
    if sd_models.model_data.sd_model is None:
        print('Error: CLIP called before SD is loaded!')
        return

    memory_management.load_model_gpu(sd_models.model_data.sd_model.forge_objects.clip.patcher)
    return

