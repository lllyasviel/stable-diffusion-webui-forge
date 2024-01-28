
def initialize_forge():
    from ldm_patched.modules import args_parser

    args_parser.args, _ = args_parser.parser.parse_known_args()

    import ldm_patched.modules.model_management as model_management
    import torch

    device = model_management.get_torch_device()
    torch.zeros((1, 1)).to(device, torch.float32)
    model_management.soft_empty_cache()

    import modules_forge.patch_clip
    modules_forge.patch_clip.patch_all_clip()

    import modules_forge.patch_precision
    modules_forge.patch_precision.patch_all_precision()

    import modules_forge.patch_basic
    modules_forge.patch_basic.patch_all_basics()

    if model_management.directml_enabled:
        model_management.lowvram_available = True
        model_management.OOM_EXCEPTION = Exception

    return
