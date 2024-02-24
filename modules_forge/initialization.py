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

    from ldm_patched.modules import args_parser

    args_parser.args, _ = args_parser.parser.parse_known_args()

    if args_parser.args.gpu_device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args_parser.args.gpu_device_id)
        print("Set device to:", args_parser.args.gpu_device_id)

    if args_parser.args.cuda_malloc:
        from modules_forge.cuda_malloc import try_cuda_malloc
        try_cuda_malloc()

    import ldm_patched.modules.model_management as model_management
    import torch

    monitor_module_moving()

    device = model_management.get_torch_device()
    torch.zeros((1, 1)).to(device, torch.float32)
    model_management.soft_empty_cache()

    import modules_forge.patch_basic
    modules_forge.patch_basic.patch_all_basics()

    from modules_forge import stream
    print('CUDA Stream Activated: ', stream.using_stream)

    from modules_forge.shared import diffusers_dir

    if 'TRANSFORMERS_CACHE' not in os.environ:
        os.environ['TRANSFORMERS_CACHE'] = diffusers_dir

    if 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = diffusers_dir

    if 'HF_DATASETS_CACHE' not in os.environ:
        os.environ['HF_DATASETS_CACHE'] = diffusers_dir

    if 'HUGGINGFACE_HUB_CACHE' not in os.environ:
        os.environ['HUGGINGFACE_HUB_CACHE'] = diffusers_dir

    if 'HUGGINGFACE_ASSETS_CACHE' not in os.environ:
        os.environ['HUGGINGFACE_ASSETS_CACHE'] = diffusers_dir

    if 'HF_HUB_CACHE' not in os.environ:
        os.environ['HF_HUB_CACHE'] = diffusers_dir
    return
