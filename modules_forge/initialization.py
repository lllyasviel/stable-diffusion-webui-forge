import argparse


def initialize_forge():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-vram", action="store_true")
    parser.add_argument("--normal-vram", action="store_true")
    parser.add_argument("--high-vram", action="store_true")
    parser.add_argument("--always-vram", action="store_true")
    vram_args = vars(parser.parse_known_args()[0])

    parser = argparse.ArgumentParser()
    attn_group = parser.add_mutually_exclusive_group()
    attn_group.add_argument("--attention-split", action="store_true")
    attn_group.add_argument("--attention-quad", action="store_true")
    attn_group.add_argument("--attention-pytorch", action="store_true")
    parser.add_argument("--disable-xformers", action="store_true")
    fpte_group = parser.add_mutually_exclusive_group()
    fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true")
    fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true")
    fpte_group.add_argument("--clip-in-fp16", action="store_true")
    fpte_group.add_argument("--clip-in-fp32", action="store_true")
    fp_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument("--all-in-fp32", action="store_true")
    fp_group.add_argument("--all-in-fp16", action="store_true")
    fpunet_group = parser.add_mutually_exclusive_group()
    fpunet_group.add_argument("--unet-in-bf16", action="store_true")
    fpunet_group.add_argument("--unet-in-fp16", action="store_true")
    fpunet_group.add_argument("--unet-in-fp8-e4m3fn", action="store_true")
    fpunet_group.add_argument("--unet-in-fp8-e5m2", action="store_true")
    fpvae_group = parser.add_mutually_exclusive_group()
    fpvae_group.add_argument("--vae-in-fp16", action="store_true")
    fpvae_group.add_argument("--vae-in-fp32", action="store_true")
    fpvae_group.add_argument("--vae-in-bf16", action="store_true")
    other_args = vars(parser.parse_known_args()[0])

    from ldm_patched.modules.args_parser import args

    args.always_cpu = False
    args.always_gpu = False
    args.always_high_vram = False
    args.always_low_vram = False
    args.always_no_vram = False
    args.always_offload_from_vram = True
    args.async_cuda_allocation = False
    args.disable_async_cuda_allocation = True

    if vram_args['no_vram']:
        args.always_cpu = True

    if vram_args['always_vram']:
        args.always_gpu = True

    if vram_args['high_vram']:
        args.always_offload_from_vram = False

    for k, v in other_args.items():
        setattr(args, k, v)

    import ldm_patched.modules.model_management as model_management
    import torch

    device = model_management.get_torch_device()
    torch.zeros((1, 1)).to(device, torch.float32)
    model_management.soft_empty_cache()
    return
