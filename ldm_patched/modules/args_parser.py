import argparse
import enum
import ldm_patched.modules.options

class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)
        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")
        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")
        super(EnumAction, self).__init__(**kwargs)
        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)

parser = argparse.ArgumentParser()
#parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0")
#parser.add_argument("--port", type=int, default=8188)
parser.add_argument("--disable-header-check", type=str, default=None, metavar="ORIGIN", nargs="?", const="*", help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
parser.add_argument("--web-upload-size", type=float, default=100, help="Set the maximum upload size in MB.")
parser.add_argument("--external-working-path", type=str, default=None, metavar="PATH", nargs='+', action='append')
parser.add_argument("--output-path", type=str, default=None)
parser.add_argument("--temp-path", type=str, default=None,)
parser.add_argument("--cache-path", type=str, default=None)
parser.add_argument("--in-browser", action="store_true")
parser.add_argument("--disable-in-browser", action="store_true", help="Disable auto launching the browser.")
parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID", help="Set the id of the cuda device this instance will use.")
parser.add_argument("--disable-attention-upcast", action="store_true", help="Disable all upcasting of attention. Should be unnecessary except for debugging.")

# New arguments
# parser.add_argument("--tls-keyfile", type=str, help="Path to TLS (SSL) key file. Enables TLS, makes app accessible at https://... requires --tls-certfile to function")
# parser.add_argument("--tls-certfile", type=str, help="Path to TLS (SSL) certificate file. Enables TLS, makes app accessible at https://... requires --tls-keyfile to function")
parser.add_argument("--input-directory", type=str, default=None)

cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument("--disable-cuda-malloc", action="store_true", help="Disable cudaMallocAsync.")

parser.add_argument("--force-channels-last", action="store_true", help="Force channels last format when inferencing the models.")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--all-in-fp32", action="store_true", help="Force fp32 (If this makes your GPU work better please report it).")
fp_group.add_argument("--all-in-fp16", action="store_true", help="Force fp16.")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--unet-in-bf16", action="store_true", help="Run the UNET in bf16. This should only be used for testing stuff.")
fpunet_group.add_argument("--unet-in-fp16", action="store_true", help="Store unet weights in fp16.")
fpunet_group.add_argument("--fp32-unet", action="store_true", help="Run the diffusion model in fp32.")
fpunet_group.add_argument("--fp64-unet", action="store_true", help="Run the diffusion model in fp64.")
fpunet_group.add_argument("--unet-in-fp8-e4m3fn", action="store_true", help="Store unet weights in fp8_e4m3fn.")
fpunet_group.add_argument("--unet-in-fp8-e5m2", action="store_true", help="Store unet weights in fp8_e5m2.")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--vae-in-fp16", action="store_true", help="Run the VAE in fp16, might cause black images.")
fpvae_group.add_argument("--vae-in-fp32", action="store_true", help="Run the VAE in full precision fp32.")
fpvae_group.add_argument("--vae-in-bf16", action="store_true", help="Run the VAE in bf16.")
parser.add_argument("--vae-in-cpu", action="store_true", help="Run the VAE on the CPU.")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true", help="Store text encoder weights in fp8 (e4m3fn variant).")
fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true", help="Store text encoder weights in fp8 (e5m2 variant).")
fpte_group.add_argument("--clip-in-fp16", action="store_true", help="Store text encoder weights in fp16.")
fpte_group.add_argument("--clip-in-fp32", action="store_true", help="Store text encoder weights in fp32.")

parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1, help="Use torch-directml.")
parser.add_argument("--disable-ipex-hijack", action="store_true", help="Disables ipex.optimize when loading models with Intel GPUs.")

class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "fast"
    TAESD = "taesd"

parser.add_argument("--preview-option", type=LatentPreviewMethod, default=LatentPreviewMethod.NoPreviews, action=EnumAction, help="Default preview method for sampler nodes.")

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--attention-split", action="store_true", help="Use the split cross attention optimization. Ignored when xformers is used.")
attn_group.add_argument("--attention-quad", action="store_true", help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.")
attn_group.add_argument("--attention-pytorch", action="store_true", help="Use the new pytorch 2.0 cross attention function.")
parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

upcast = parser.add_mutually_exclusive_group()
upcast.add_argument("--force-upcast-attention", action="store_true", help="Force enable attention upcasting, please report if it fixes black images.")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--always-gpu", action="store_true", help="Store and run everything (text encoders/CLIP models, etc... on the GPU).")
vram_group.add_argument("--always-high-vram", action="store_true", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
vram_group.add_argument("--always-normal-vram", action="store_true", help="Used to force normal vram use if lowvram gets automatically enabled.")
vram_group.add_argument("--always-low-vram", action="store_true", help="Split the unet in parts to use less vram.")
vram_group.add_argument("--always-no-vram", action="store_true", help="When lowvram isn't enough.")
vram_group.add_argument("--always-cpu", action="store_true", help="To use the CPU for everything (slow).")

parser.add_argument("--reserve-vram", type=float, default=None, help="Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reverved depending on your OS.")

parser.add_argument("--always-offload-from-vram", action="store_true", help="Force reForge to agressively offload to regular ram instead of keeping models in vram when it can.")
parser.add_argument("--pytorch-deterministic", action="store_true", help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.")
parser.add_argument("--fast", action="store_true", help="Enable some untested and potentially quality deteriorating optimizations.")
parser.add_argument("--disable-server-log", action="store_true", help="Don't print server output.")
parser.add_argument("--debug-mode", action="store_true", help="Enables more debug prints.")
parser.add_argument("--is-windows-embedded-python", action="store_true", help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup).")
parser.add_argument("--disable-server-info", action="store_true", help="Disable saving prompt metadata in files.")
parser.add_argument("--multi-user", action="store_true", help="Enables per-user storage.")
parser.add_argument("--cuda-malloc", action="store_true")
parser.add_argument("--cuda-stream", action="store_true")
parser.add_argument("--pin-shared-memory", action="store_true")

# New arguments
parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")
parser.add_argument("--disable-all-custom-nodes", action="store_true", help="Disable loading all custom nodes.")
parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")

if ldm_patched.modules.options.args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

if args.is_windows_embedded_python:
    args.in_browser = True
if args.disable_in_browser:
    args.in_browser = False

import logging
logging_level = logging.INFO
if args.debug_mode:
    logging_level = logging.DEBUG
    logging.basicConfig(format="%(message)s", level=logging_level)