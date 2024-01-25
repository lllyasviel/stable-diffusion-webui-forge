
def initialize_forge():
    import ldm_patched.modules.args_parser as args_parser

    args_parser.parser.add_argument("--disable-offload-from-vram", action="store_true",
                                    help="Force loading models to vram when the unload can be avoided. "
                                         "Use this when you ara on MAC or have more than 20GB VRAM like RTX4096.")

    args_parser.args = args_parser.parser.parse_known_args()[0]
    args_parser.args.always_offload_from_vram = not args_parser.args.disable_offload_from_vram

    import ldm_patched.modules.model_management as model_management
    import torch

    device = model_management.get_torch_device()
    torch.zeros((1, 1)).to(device, torch.float32)
    model_management.soft_empty_cache()

    import modules_forge.patch_clip
    modules_forge.patch_clip.patch_all_clip()

    import modules_forge.patch_precision
    modules_forge.patch_precision.patch_all_precision()

    return
