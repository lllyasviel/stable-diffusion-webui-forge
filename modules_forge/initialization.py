import os
import sys


def initialize_forge():
    bad_list = ['--lowvram', '--medvram', '--medvram-sdxl']

    for bad in bad_list:
        if bad in sys.argv:
            print(f'Arg {bad} is removed in Forge.')
            print(f'Now memory management is fully automatic and you do not need any command flags.')
            print(f'Please just remove this flag.')
            print(f'In extreme cases, if you want to force previous lowvram/medvram behaviors, '
                  f'please use --always-offload-from-vram')
            exit(0)

    from ldm_patched.modules import args_parser

    args_parser.args, _ = args_parser.parser.parse_known_args()

    import ldm_patched.modules.model_management as model_management
    import torch

    device = model_management.get_torch_device()
    torch.zeros((1, 1)).to(device, torch.float32)
    model_management.soft_empty_cache()

    import ldm_patched.utils.path_utils as ldm_patched_path_utils
    from modules.paths import models_path, data_path

    ldm_patched_path_utils.base_path = data_path
    ldm_patched_path_utils.models_dir = models_path
    ldm_patched_path_utils.output_directory = os.path.join(data_path, "output")
    ldm_patched_path_utils.temp_directory = os.path.join(data_path, "temp")
    ldm_patched_path_utils.input_directory = os.path.join(data_path, "input")
    ldm_patched_path_utils.user_directory = os.path.join(data_path, "user")

    import modules_forge.patch_basic
    modules_forge.patch_basic.patch_all_basics()

    from modules_forge import supported_preprocessor
    from modules_forge import supported_controlnet

    return
