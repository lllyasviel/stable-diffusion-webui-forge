import os
import sys


INITIALIZED = False
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
    global INITIALIZED

    if INITIALIZED:
        return

    INITIALIZED = True

    # Import timer for profiling
    from modules import timer
    startup_timer = timer.startup_timer

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'packages_3rdparty'))

    bad_list = ['--lowvram', '--medvram', '--medvram-sdxl']

    for bad in bad_list:
        if bad in sys.argv:
            print(f'Arg {bad} is removed in Forge.')
            print('Now memory management is fully automatic and you do not need any command flags.')
            print('Please just remove this flag.')
            print('In extreme cases, if you want to force previous lowvram/medvram behaviors, '
                  'please use --always-offload-from-vram')

    with startup_timer.subcategory("device setup"):
        from backend.args import args
        
        if args.gpu_device_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
            print("Set device to:", args.gpu_device_id)
        startup_timer.record("GPU device selection")
        
        if args.cuda_malloc:
            from modules_forge.cuda_malloc import try_cuda_malloc
            try_cuda_malloc()
        startup_timer.record("CUDA malloc setup")

    with startup_timer.subcategory("memory management init"):
        from backend import memory_management
        import torch
        startup_timer.record("memory management import")

        monitor_module_moving()

        # Device detection and CUDA context creation
        device = memory_management.get_torch_device()
        startup_timer.record("device detection")
        
        # CUDA context initialization
        torch.zeros((1, 1)).to(device, torch.float32)
        memory_management.soft_empty_cache()
        startup_timer.record("CUDA context creation")
        
        # Print detailed memory management summary for debugging
        memory_management.print_memory_management_summary()
        startup_timer.record("memory management analysis")

    with startup_timer.subcategory("backend capabilities"):
        # Quantization support detection
        if memory_management.can_install_bnb():
            from modules_forge.bnb_installer import try_install_bnb
            try_install_bnb()
        startup_timer.record("quantization support")

        # Stream capabilities
        from backend import stream
        print('CUDA Using Stream:', stream.should_use_stream())
        startup_timer.record("stream capabilities")

    with startup_timer.subcategory("environment setup"):
        from modules_forge.shared import diffusers_dir

        # if 'TRANSFORMERS_CACHE' not in os.environ:
        #     os.environ['TRANSFORMERS_CACHE'] = diffusers_dir

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
        startup_timer.record("environment variables")

    # Call patch_all_basics outside the environment setup context to avoid nested category conflicts
    import modules_forge.patch_basic
    modules_forge.patch_basic.patch_all_basics()
    startup_timer.record("basic patches")

    return
