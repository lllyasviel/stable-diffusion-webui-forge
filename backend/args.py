import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")

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

parser.add_argument("--vae-in-cpu", action="store_true")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true")
fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true")
fpte_group.add_argument("--clip-in-fp16", action="store_true")
fpte_group.add_argument("--clip-in-fp32", action="store_true")

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--attention-split", action="store_true")
attn_group.add_argument("--attention-quad", action="store_true")
attn_group.add_argument("--attention-pytorch", action="store_true")
attn_group.add_argument("--use-sage-attention", action="store_true", help="Use sage attention.")
attn_group.add_argument("--use-flash-attention", action="store_true", help="Use FlashAttention.")

upcast = parser.add_mutually_exclusive_group()
upcast.add_argument("--force-upcast-attention", action="store_true")
upcast.add_argument("--disable-attention-upcast", action="store_true")

parser.add_argument("--disable-xformers", action="store_true")

parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1)
parser.add_argument("--disable-ipex-hijack", action="store_true")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--always-gpu", action="store_true")
vram_group.add_argument("--always-high-vram", action="store_true")
vram_group.add_argument("--always-normal-vram", action="store_true")
vram_group.add_argument("--always-low-vram", action="store_true")
vram_group.add_argument("--always-no-vram", action="store_true")
vram_group.add_argument("--always-cpu", action="store_true")

parser.add_argument("--always-offload-from-vram", action="store_true")
parser.add_argument("--pytorch-deterministic", action="store_true")

parser.add_argument("--cuda-malloc", action="store_true")
parser.add_argument("--cuda-stream", action="store_true")
parser.add_argument("--pin-shared-memory", action="store_true")

parser.add_argument("--disable-gpu-warning", action="store_true")

args = parser.parse_known_args()[0]

# Some dynamic args that may be changed by webui rather than cmd flags.
dynamic_args = dict(
    embedding_dir='./embeddings',
    emphasis_name='original'
)
