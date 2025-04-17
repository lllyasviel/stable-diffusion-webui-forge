import math
import sys
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Tuple

import torch
import einops

from backend import memory_management
from backend.misc.sub_quadratic_attention import efficient_dot_product_attention
from backend.operations import ForgeOperations  # ensure this import is valid

# Try optional libraries
try:
    import xformers.ops as xops
    _HAS_XFORMERS = True
except ImportError:
    _HAS_XFORMERS = False

try:
    from sageattention import sageattn
    _HAS_SAGE = True
except ImportError:
    _HAS_SAGE = False

try:
    from flash_attn import flash_attn_func
    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class AttentionConfig:
    heads: int
    disable_upcast: bool
    force_upcast_dtype: Optional[torch.dtype]
    device: torch.device


def _get_upcast_dtype(cfg: AttentionConfig, default: torch.dtype) -> Optional[torch.dtype]:
    if cfg.disable_upcast:
        return None
    return cfg.force_upcast_dtype or default


def _reshape_qkv(
    t: torch.Tensor, batch: int, heads: int, dim_head: int, skip: bool
) -> torch.Tensor:
    """Reshape a [B, T, H*D] or [B, H, T, D] tensor into [B*H, T, D]."""
    if skip:
        return t.reshape(batch * heads, -1, dim_head)
    # from [B, T, H*D] → [B, T, H, D] → [B, H, T, D] → [B*H, T, D]
    return (
        t.unsqueeze(-1)
         .reshape(batch, -1, heads, dim_head)
         .permute(0, 2, 1, 3)
         .reshape(batch * heads, -1, dim_head)
         .contiguous()
    )


def _unshape_out(
    out: torch.Tensor, batch: int, heads: int, dim_head: int, skip: bool
) -> torch.Tensor:
    """Reverse of `_reshape_qkv` for the output of cross‑attention."""
    if skip:
        return (
            out
            .unsqueeze(0)
            .reshape(batch, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(batch, -1, heads * dim_head)
        )
    return out.reshape(batch, -1, heads * dim_head)


def _prepare_mask(
    mask: Optional[torch.Tensor], batch: int, heads: int
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    # expected mask shape: [B, ..., T, T]
    bsz = mask.shape[0] if mask.ndim > 2 else 1
    m = mask.reshape(bsz, -1, mask.shape[-2], mask.shape[-1])
    m = m.expand(batch, heads, -1, -1)
    return m.reshape(-1, m.shape[-2], m.shape[-1])


# Attention backend type
AttentionFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, AttentionConfig, Optional[torch.Tensor], bool],
    torch.Tensor,
]

_ATTENTION_REGISTRY: Dict[str, AttentionFn] = {}


def register_attention(name: str):
    def decorator(fn: AttentionFn):
        _ATTENTION_REGISTRY[name] = fn
        return fn
    return decorator


@register_attention("basic")
def attention_basic(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: AttentionConfig,
    mask: Optional[torch.Tensor] = None,
    skip_reshape: bool = False,
) -> torch.Tensor:
    """
    Standard O(N^2) scaled-dot-product attention.
    """
    upcast = _get_upcast_dtype(cfg, torch.float32)
    batch = q.shape[0]
    dim_head = q.shape[-1] // (1 if skip_reshape else cfg.heads)
    scale = dim_head ** -0.5

    # reshape
    q_ = _reshape_qkv(q, batch, cfg.heads, dim_head, skip_reshape)
    k_ = _reshape_qkv(k, batch, cfg.heads, dim_head, skip_reshape)
    v_ = _reshape_qkv(v, batch, cfg.heads, dim_head, skip_reshape)

    # dot-product
    if upcast is torch.float32:
        sim = torch.einsum("b i d, b j d -> b i j", q_.float(), k_.float()) * scale
    else:
        sim = torch.einsum("b i d, b j d -> b i j", q_, k_) * scale

    # mask
    m = _prepare_mask(mask, batch, cfg.heads)
    if m is not None:
        sim = sim.masked_fill(~m.bool(), float("-inf"))

    # softmax + gather
    attn = sim.softmax(dim=-1)
    out = torch.einsum("b i j, b j d -> b i d", attn.to(v_.dtype), v_)

    # unshape
    return _unshape_out(out, batch, cfg.heads, dim_head, skip_reshape)


@register_attention("subquadratic")
def attention_subquadratic(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: AttentionConfig,
    mask: Optional[torch.Tensor] = None,
    skip_reshape: bool = False,
) -> torch.Tensor:
    """
    Memory‑efficient subquadratic attention via chunking.
    """
    upcast = _get_upcast_dtype(cfg, torch.float32)
    batch = q.shape[0]
    dim_head = q.shape[-1] // (1 if skip_reshape else cfg.heads)
    scale = dim_head ** -0.5

    # reshape & transpose key for subquadratic API
    q_ = _reshape_qkv(q, batch, cfg.heads, dim_head, skip_reshape)
    v_ = _reshape_qkv(v, batch, cfg.heads, dim_head, skip_reshape)
    k_ = _reshape_qkv(k, batch, cfg.heads, dim_head, skip_reshape).movedim(1, 2)

    # prepare mask
    m = _prepare_mask(mask, batch, cfg.heads)

    # efficient kernel
    out = efficient_dot_product_attention(
        q_, k_, v_,
        query_chunk_size=None,  # let the kernel decide or you can tune
        kv_chunk_size=None,
        kv_chunk_size_min=None,
        use_checkpoint=False,
        upcast_attention=(upcast is torch.float32),
        mask=m,
    ).to(q_.dtype)

    # back to [B, T, H*D]
    return _unshape_out(out, batch, cfg.heads, dim_head, skip_reshape)


@register_attention("xformers")
def attention_xformers(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: AttentionConfig,
    mask: Optional[torch.Tensor] = None,
    skip_reshape: bool = False,
) -> torch.Tensor:
    """
    xFormers memory‑efficient attention.
    """
    if not _HAS_XFORMERS:
        logging.warning("xFormers not installed, falling back to basic attention")
        return attention_basic(q, k, v, cfg, mask, skip_reshape)

    batch = q.shape[0]
    dim_head = q.shape[-1] // (1 if skip_reshape else cfg.heads)

    # reshape just enough for xformers
    q_ = _reshape_qkv(q, batch, cfg.heads, dim_head, skip_reshape)
    k_ = _reshape_qkv(k, batch, cfg.heads, dim_head, skip_reshape)
    v_ = _reshape_qkv(v, batch, cfg.heads, dim_head, skip_reshape)

    m = _prepare_mask(mask, batch, cfg.heads)
    out = xops.memory_efficient_attention(q_, k_, v_, attn_bias=m)
    return _unshape_out(out, batch, cfg.heads, dim_head, skip_reshape)


@register_attention("flash")
def attention_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: AttentionConfig,
    mask: Optional[torch.Tensor] = None,
    skip_reshape: bool = False,
) -> torch.Tensor:
    """
    FlashAttention via torch.library.custom_op wrapper.
    """
    if not _HAS_FLASH:
        logging.warning("flash-attn not installed, falling back to basic attention")
        return attention_basic(q, k, v, cfg, mask, skip_reshape)

    # use PyTorch's scaled_dot_product to avoid repeated reshape logic
    B, T, _ = q.shape if skip_reshape else q.view(q.shape[0], -1, cfg.heads, cfg.heads).transpose(1, 2).shape
    q_, k_, v_ = (q, k, v)  # assume q,k,v already in [B, H, T, D] for flash
    m = None  # flash wrapper currently ignores mask
    out = flash_attn_wrapper(q_, k_, v_, 0.0, False)
    return _unshape_out(out, q.shape[0], cfg.heads, q.shape[-1] // cfg.heads, skip_reshape)


def get_attention_fn(name: str) -> AttentionFn:
    try:
        return _ATTENTION_REGISTRY[name]
    except KeyError:
        raise ValueError(f"No attention backend named '{name}' registered.")


# Finally, in your module initialization you pick one:
def select_attention_backend() -> AttentionFn:
    if memory_management.sage_attention_enabled() and _HAS_SAGE:
        logging.info("Using SageAttention backend")
        return get_attention_fn("sage")  # you’d register it similarly
    if memory_management.xformers_enabled():
        logging.info("Using xFormers backend")
        return get_attention_fn("xformers")
    if memory_management.flash_attention_enabled():
        logging.info("Using FlashAttention backend")
        return get_attention_fn("flash")
    if memory_management.pytorch_attention_enabled():
        logging.info("Using PyTorch SDPA backend")
        return get_attention_fn("basic")
    if args.attention_split:
        logging.info("Using sub-quadratic backend")
        return get_attention_fn("subquadratic")
    logging.info("Defaulting to basic attention")
    return get_attention_fn("basic")


# You’d then do:
AttentionFunction = select_attention_backend()

# And in your ForgeProcessor:
class AttentionProcessorForge:
    def __call__(
        self,
        attn,  # your module with to_q, to_k, to_v, etc.
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args, **kwargs
    ) -> torch.Tensor:
        cfg = AttentionConfig(
            heads=attn.heads,
            disable_upcast=args.disable_attention_upcast,
            force_upcast_dtype=memory_management.force_upcast_attention_dtype(),
            device=hidden_states.device,
        )
        # ... same preprocessing of spatial_norm, group_norm, reshape to 2D, etc. ...
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states or hidden_states)
        v = attn.to_v(encoder_hidden_states or hidden_states)

        out = AttentionFunction(q, k, v, cfg, attention_mask, skip_reshape=False)
        # ... postprocessing via attn.to_out and residuals ...
        return out
