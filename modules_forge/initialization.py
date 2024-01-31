import os
import sys


MONITOR_MODEL_MOVING = False


def monitor_module_moving():
    if not MONITOR_MODEL_MOVING:
        return

    import torch
    import traceback

    old_to = torch.nn.Module.to

    def new_to(*args, **kwargs):
        traceback.print_stack()
        print('Model Movement')

        return old_to(*args, **kwargs)

    torch.nn.Module.to = new_to
    return


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

    monitor_module_moving()

    device = model_management.get_torch_device()
    torch.zeros((1, 1)).to(device, torch.float32)
    model_management.soft_empty_cache()

    import modules_forge.patch_basic
    modules_forge.patch_basic.patch_all_basics()

    from modules_forge import supported_preprocessor
    from modules_forge import supported_controlnet

    return
