import argparse
import json
import logging
import os

def setup_logging(verbosity: int):
    level = max(logging.DEBUG, logging.WARNING - 10 * verbosity)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced GPU/precision configuration for ML inference"
    )
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file with argument defaults",
    )

    # Hardware
    hw = parser.add_argument_group("Hardware Configuration")
    hw.add_argument(
        "--gpu-device-id",
        type=int,
        default=None,
        metavar="ID",
        help="Index of GPU to use (None = CPU)",
    )
    hw.add_argument(
        "--directml",
        nargs="?",
        const=-1,
        type=int,
        metavar="DEVICE",
        help="Enable DirectML on given device (default: all devices)",
    )

    # Global precision
    prec = parser.add_argument_group("Precision Settings")
    prec.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp16",
        help="Global precision for all modules",
    )
    prec.add_argument(
        "--unet-precision",
        choices=["bf16", "fp16", "fp8-e4m3fn", "fp8-e5m2"],
        help="Precision override for UNet",
    )
    prec.add_argument(
        "--vae-precision",
        choices=["bf16", "fp16", "fp32"],
        help="Precision override for VAE",
    )
    prec.add_argument(
        "--vae-cpu",
        action="store_true",
        help="Force VAE to load on CPU",
    )

    # Attention
    attn = parser.add_argument_group("Attention Options")
    attn.add_argument(
        "--attention",
        choices=["split", "quad", "pytorch", "sage", "flash"],
        default="pytorch",
        help="Which attention implementation to use",
    )
    attn.add_argument(
        "--upcast-attention",
        action="store_true",
        help="Force upcast inside attention kernels",
    )
    attn.add_argument(
        "--disable-attention-upcast",
        action="store_true",
        help="Prevent any upcasting in attention",
    )

    # VRAM policy
    parser.add_argument(
        "--vram-mode",
        choices=["always-gpu", "high", "normal", "low", "no-vram", "cpu"],
        default="normal",
        help="VRAM usage policy (overrides other vram flags)",
    )
    parser.add_argument(
        "--always-offload-from-vram",
        action="store_true",
        help="Offload tensors from VRAM when idle",
    )

    # Performance tweaks
    perf = parser.add_argument_group("Performance Tweaks")
    perf.add_argument(
        "--pytorch-deterministic",
        action="store_true",
        help="Enable deterministic mode in PyTorch",
    )
    perf.add_argument(
        "--cuda-malloc",
        action="store_true",
        help="Use cudaMalloc for allocations",
    )
    perf.add_argument(
        "--cuda-stream",
        action="store_true",
        help="Enable CUDA streams",
    )
    perf.add_argument(
        "--pin-shared-memory",
        action="store_true",
        help="Pin shared memory for faster transfers",
    )
    perf.add_argument(
        "--disable-xformers",
        action="store_true",
        help="Disable XFormers optimizations",
    )
    perf.add_argument(
        "--disable-ipex-hijack",
        action="store_true",
        help="Disable Intel IPEX hijacking",
    )
    perf.add_argument(
        "--disable-gpu-warning",
        action="store_true",
        help="Suppress GPU compatibility warnings",
    )

    # Debug / utility
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (repeat for more)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print parsed arguments and exit",
    )

    # First pass: load CLI + config file
    args = parser.parse_args()
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            cfg = json.load(f)
        parser.set_defaults(**cfg)
        args = parser.parse_args()  # reparse with config defaults

    # Validation
    if args.upcast_attention and args.disable_attention_upcast:
        parser.error("Cannot use both --upcast-attention and --disable-attention-upcast")

    setup_logging(args.verbose)
    return args

def main():
    args = parse_args()
    if args.dry_run:
        print(args)
        return

    logging.info("Starting with configuration: %s", args)
    # ... rest of your application logic ...

if __name__ == "__main__":
    main()
