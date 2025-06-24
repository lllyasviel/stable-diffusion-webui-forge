from typing import Optional, Tuple, Union, Any, Callable, Dict
from functools import wraps
import torch
import torch.nn.functional as F
from torch import Tensor
import math
import warnings
import einops
from contextlib import suppress
from backend import memory_management
from backend.misc.sub_quadratic_attention import efficient_dot_product_attention

def get_attn_precision(attn_precision: Optional[torch.dtype] = None) -> torch.dtype:
    """Get the attention computation precision
    
    Args:
        attn_precision: Optional dtype override. If None, uses the default precision.
        
    Returns:
        torch.dtype: The dtype to use for attention computation
    """
    if attn_precision is not None:
        return attn_precision
        
    # Check if we should force upcast
    if memory_management.force_upcast_attention_dtype():
        return torch.float32
        
    # Default to float16 for efficiency if not forcing upcast
    return torch.float16

# Constants and global state
XFORMERS_AVAILABLE = False
FLASH_ATTENTION_AVAILABLE = False
BROKEN_XFORMERS = False

# Safe module-level imports with proper typing
xformers = None  # type: Any
flash_attn_func = None  # type: Any

class AttentionError(Exception):
    """Custom exception for attention-related errors"""
    pass

def safe_import_xformers() -> None:
    """Safely import xformers with proper error handling"""
    global XFORMERS_AVAILABLE, BROKEN_XFORMERS, xformers
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import xformers
            import xformers.ops
        XFORMERS_AVAILABLE = True
        
        # Check for broken versions
        try:
            version = getattr(xformers, '__version__', None)
            if version and version.startswith("0.0.2") and not version.startswith("0.0.20"):
                BROKEN_XFORMERS = True
                warnings.warn(f"Detected broken xformers version: {version}")
        except Exception:
            pass
            
    except ImportError:
        pass

def safe_import_flash_attention() -> None:
    """Safely import Flash Attention with proper error handling"""
    global FLASH_ATTENTION_AVAILABLE, flash_attn_func
    
    try:
        from flash_attn import flash_attn_func
        FLASH_ATTENTION_AVAILABLE = True
    except ImportError:
        pass

def initialize_attention_backends() -> None:
    """Initialize all attention backends"""
    safe_import_xformers()
    safe_import_flash_attention()

initialize_attention_backends()

def process_mask(
    mask: Tensor,
    batch_size: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device
) -> Tensor:
    """
    Process attention mask into the correct format
    
    Args:
        mask: Input mask tensor
        batch_size: Batch size
        num_heads: Number of attention heads
        dtype: Desired dtype
        device: Target device
    
    Returns:
        Processed mask tensor
    """
    if mask.dtype == torch.bool:
        mask = mask.logical_not()
        mask = mask.to(dtype)
        mask = torch.where(mask, -torch.finfo(dtype).max, 0.0)
    
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    
    # Expand mask for all heads if needed
    if mask.dim() == 3:
        mask = mask.unsqueeze(1).expand(batch_size, num_heads, *mask.shape[1:])
    
    return mask.to(device=device, dtype=dtype)

def validate_attention_shapes(q: Tensor, k: Tensor, v: Tensor) -> None:
    """Validate shapes of attention inputs"""
    if not (q.dim() == k.dim() == v.dim()):
        raise AttentionError("Query, key and value must have same number of dimensions")
        
    if not (q.size(-1) == k.size(-1)):
        raise AttentionError("Query and key must have same embedding dimension")
        
    if not (k.size(-2) == v.size(-2)):
        raise AttentionError("Key and value must have same sequence length")

def get_attention_backend() -> str:
    """Get the most efficient available attention backend"""
    if XFORMERS_AVAILABLE and not BROKEN_XFORMERS:
        return "xformers"
    elif FLASH_ATTENTION_AVAILABLE:
        return "flash"
    elif memory_management.pytorch_attention_enabled():
        return "pytorch"
    else:
        return "chunked"

def compute_attention_chunked(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    chunk_size: int,
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Compute attention scores in chunks to save memory
    
    Args:
        q: Query tensor
        k: Key tensor  
        v: Value tensor
        scale: Attention scale factor
        chunk_size: Size of chunks to process
        mask: Optional attention mask
        
    Returns:
        Output tensor
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    chunks = []
    
    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)
        
        # Get current chunk
        q_chunk = q[:, :, start_idx:end_idx]
        
        # Compute attention scores for chunk
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            if mask.dim() == 4:
                chunk_mask = mask[:, :, start_idx:end_idx]
            else:
                chunk_mask = mask[:, None, start_idx:end_idx]
            scores = scores + chunk_mask
            
        # Compute attention weights and apply to values
        attn = F.softmax(scores, dim=-1)
        chunks.append(torch.matmul(attn, v))
        
    return torch.cat(chunks, dim=2)

def memory_efficient_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    upcast: bool = True,
    recompute: bool = False
) -> Tensor:
    """
    Memory efficient attention implementation with multiple backend options
    
    Args:
        q: Query tensor [batch_size, seq_len, dim]
        k: Key tensor [batch_size, seq_len, dim]  
        v: Value tensor [batch_size, seq_len, dim]
        heads: Number of attention heads
        mask: Optional attention mask
        dropout_p: Dropout probability
        upcast: Whether to automatically upcast to higher precision
        recompute: Whether to recompute attention to save memory
        
    Returns:
        Output tensor [batch_size, seq_len, dim]
    """
    # Validate inputs
    validate_attention_shapes(q, k, v)
    
    # Get computation dtype
    dtype = q.dtype
    compute_dtype = get_attn_precision() if upcast else dtype
    
    # Process inputs
    batch_size = q.size(0) 
    seq_len = q.size(1)
    head_dim = q.size(-1) // heads
    scaling = head_dim ** -0.5
    
    # Reshape into multiple heads
    q = q.reshape(batch_size, seq_len, heads, head_dim)
    k = k.reshape(batch_size, k.size(1), heads, head_dim)
    v = v.reshape(batch_size, v.size(1), heads, head_dim)
    
    # Process mask if provided
    if mask is not None:
        mask = process_mask(mask, batch_size, heads, compute_dtype, q.device)
    
    # Try different backends in order of efficiency
    backend = get_attention_backend()
    
    try:
        if backend == "xformers" and xformers is not None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            if compute_dtype != dtype:
                q = q.to(compute_dtype)
                k = k.to(compute_dtype)
                v = v.to(compute_dtype)
                
            out = xformers.ops.memory_efficient_attention(
                q, k, v,
                attn_bias=mask,
                p=dropout_p
            )
            
        elif backend == "flash" and flash_attn_func is not None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            if compute_dtype != dtype:
                q = q.to(compute_dtype)
                k = k.to(compute_dtype)
                v = v.to(compute_dtype)
                
            out = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=scaling,
                causal=False
            )
            
        elif backend == "pytorch" and hasattr(F, 'scaled_dot_product_attention'):
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            if compute_dtype != dtype:
                q = q.to(compute_dtype)
                k = k.to(compute_dtype)
                v = v.to(compute_dtype)
                
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=dropout_p,
                scale=scaling
            )
            
        else:
            # Fall back to chunked implementation
            if compute_dtype != dtype:
                q = q.to(compute_dtype)
                k = k.to(compute_dtype)
                v = v.to(compute_dtype)
                
            # Transpose for attention computation
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Compute attention in chunks
            chunk_size = min(128, seq_len)
            out = compute_attention_chunked(
                q, k, v,
                scale=scaling,
                chunk_size=chunk_size,
                mask=mask
            )
            
            if dropout_p > 0:
                out = F.dropout(out, p=dropout_p, training=True)
                
        # Convert back to original dtype and shape
        if compute_dtype != dtype:
            out = out.to(dtype)
            
        out = out.transpose(1, 2)
        
    except Exception as e:
        raise AttentionError(f"Attention computation failed: {str(e)}")
        
    # Return reshaped output
    return out.reshape(batch_size, seq_len, heads * head_dim)

# Set as primary attention implementation
attention_function = memory_efficient_attention

def exists(val: Any) -> bool:
    """Check if a value exists (is not None)"""
    return val is not None

def get_attention_precision(attn_precision: Optional[torch.dtype] = None) -> torch.dtype:
    """Get attention computation precision"""
    if attn_precision is None:
        return torch.float32
    return attn_precision

def xformers_attention_single_head_spatial(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """xFormers implementation for single head spatial attention"""
    if not XFORMERS_AVAILABLE:
        return pytorch_attention_single_head_spatial(q, k, v)
        
    try:
        B, C, H, W = q.shape
        q, k, v = map(
            lambda t: t.reshape(B, C, -1).transpose(1, 2).contiguous(),
            (q, k, v),
        )

        if XFORMERS_AVAILABLE:
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        else:
            scale = math.sqrt(C)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out
    except Exception as e:
        warnings.warn(f"xFormers spatial attention failed: {e}")
        return pytorch_attention_single_head_spatial(q, k, v)

def pytorch_attention_single_head_spatial(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """PyTorch implementation for single head spatial attention"""
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.reshape(B, C, -1),
        (q, k, v),
    )

    # Scaled dot product attention
    scale = math.sqrt(C)
    scores = torch.matmul(q.transpose(1, 2), k) / scale
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.transpose(1, 2))
    
    return out.transpose(1, 2).reshape(B, C, H, W)

def slice_attention_single_head_spatial(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Memory efficient sliced implementation for single head spatial attention"""
    B, C, H, W = q.shape
    q = q.reshape(B, C, -1).transpose(1, 2)  # B, HW, C
    k = k.reshape(B, C, -1)  # B, C, HW
    v = v.reshape(B, C, -1)  # B, C, HW
    
    scale = math.sqrt(C)
    scores = torch.matmul(q, k) / scale  # B, HW, HW
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.transpose(1, 2))  # B, HW, C
    
    return out.transpose(1, 2).reshape(B, C, H, W)

def normal_attention_single_head_spatial(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Standard implementation for single head spatial attention"""
    return slice_attention_single_head_spatial(q, k, v)

# Set default single head spatial attention implementation
if memory_management.xformers_enabled_vae():
    print("Using xformers attention for VAE")
    attention_function_single_head_spatial = xformers_attention_single_head_spatial
elif memory_management.pytorch_attention_enabled():
    print("Using pytorch attention for VAE")
    attention_function_single_head_spatial = pytorch_attention_single_head_spatial
else:
    print("Using split attention for VAE")
    attention_function_single_head_spatial = normal_attention_single_head_spatial


def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    attn_precision = get_attention_precision(attn_precision)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    scale = dim_head ** -0.5

    h = heads
    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

    if attn_precision == torch.float32:
        sim = torch.einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if exists(mask):
        if mask.dtype == torch.bool:
            mask = einops.rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = einops.repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        else:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
            sim.add_(mask)

    sim = sim.softmax(dim=-1)
    out = torch.einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return out


def attention_sub_quad(query, key, value, heads, mask=None, attn_precision=None, skip_reshape=False):
    attn_precision = get_attention_precision(attn_precision)

    if skip_reshape:
        b, _, _, dim_head = query.shape
    else:
        b, _, dim_head = query.shape
        dim_head //= heads

    scale = dim_head ** -0.5

    if skip_reshape:
        query = query.reshape(b * heads, -1, dim_head)
        value = value.reshape(b * heads, -1, dim_head)
        key = key.reshape(b * heads, -1, dim_head).movedim(1, 2)
    else:
        query = query.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
        value = value.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
        key = key.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 3, 1).reshape(b * heads, dim_head, -1)

    dtype = query.dtype
    upcast_attention = attn_precision == torch.float32 and query.dtype != torch.float32
    if upcast_attention:
        bytes_per_token = torch.finfo(torch.float32).bits // 8
    else:
        bytes_per_token = torch.finfo(query.dtype).bits // 8
    batch_x_heads, q_tokens, _ = query.shape
    _, _, k_tokens = key.shape
    qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

    mem_free_total, mem_free_torch = memory_management.get_free_memory(query.device, True)

    kv_chunk_size_min = None
    kv_chunk_size = None
    query_chunk_size = None

    for x in [4096, 2048, 1024, 512, 256]:
        count = mem_free_total / (batch_x_heads * bytes_per_token * x * 4.0)
        if count >= k_tokens:
            kv_chunk_size = k_tokens
            query_chunk_size = x
            break

    if query_chunk_size is None:
        query_chunk_size = 512

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

    hidden_states = efficient_dot_product_attention(
        query,
        key,
        value,
        query_chunk_size=query_chunk_size,
        kv_chunk_size=kv_chunk_size,
        kv_chunk_size_min=kv_chunk_size_min,
        use_checkpoint=False,
        upcast_attention=upcast_attention,
        mask=mask,
    )

    hidden_states = hidden_states.to(dtype)

    hidden_states = hidden_states.unflatten(0, (-1, heads)).transpose(1, 2).flatten(start_dim=2)
    return hidden_states


def attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    attn_precision = get_attention_precision(attn_precision)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    scale = dim_head ** -0.5

    h = heads
    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

    mem_free_total = memory_management.get_free_memory(q.device)

    if attn_precision == torch.float32:
        element_size = 4
        upcast = True
    else:
        element_size = q.element_size()
        upcast = False

    gb = 1024 ** 3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * element_size
    modifier = 3
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
        # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
        #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

    if steps > 64:
        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
        raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                           f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

    # print("steps", steps, mem_required, mem_free_total, modifier, q.element_size(), tensor_size)
    first_op_done = False
    cleared_cache = False
    while True:
        try:
            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                if upcast:
                    with torch.autocast(enabled=False, device_type='cuda'):
                        s1 = torch.einsum('b i d, b j d -> b i j', q[:, i:end].float(), k.float()) * scale
                else:
                    s1 = torch.einsum('b i d, b j d -> b i j', q[:, i:end], k) * scale

                if mask is not None:
                    if len(mask.shape) == 2:
                        s1 += mask[i:end]
                    else:
                        s1 += mask[:, i:end]

                s2 = s1.softmax(dim=-1).to(v.dtype)
                del s1
                first_op_done = True

                r1[:, i:end] = torch.einsum('b i j, b j d -> b i d', s2, v)
                del s2
            break
        except memory_management.OOM_EXCEPTION as e:
            if first_op_done == False:
                memory_management.soft_empty_cache(True)
                if cleared_cache == False:
                    cleared_cache = True
                    print("out of memory error, emptying cache and trying again")
                    continue
                steps *= 2
                if steps > 64:
                    raise e
                print("out of memory error, increasing steps and trying again {}".format(steps))
            else:
                raise e

    del q, k, v

    r1 = (
        r1.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return r1


def attention_xformers(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    if BROKEN_XFORMERS and b * heads > 65535:
        return attention_pytorch(q, k, v, heads, mask, skip_reshape=skip_reshape)

    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.reshape(b, -1, heads, dim_head),
            (q, k, v),
        )

    if mask is not None:
        pad = 8 - q.shape[1] % 8
        mask_out = torch.empty([q.shape[0], q.shape[1], q.shape[1] + pad], dtype=q.dtype, device=q.device)
        mask_out[:, :, :mask.shape[-1]] = mask
        mask = mask_out[:, :, :mask.shape[-1]]

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    if skip_reshape:
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
    else:
        out = (
            out.reshape(b, -1, heads * dim_head)
        )

    return out


def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    """PyTorch native attention using scaled_dot_product_attention when available"""
    if hasattr(F, 'scaled_dot_product_attention'):
        if not skip_reshape:
            b, _, dim_head = q.shape
            dim_head = dim_head // heads
            q = q.reshape(b, -1, heads, dim_head).transpose(1, 2)
            k = k.reshape(b, -1, heads, dim_head).transpose(1, 2)
            v = v.reshape(b, -1, heads, dim_head).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        
        if not skip_reshape:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out
    else:
        return attention_basic(q, k, v, heads, mask, attn_precision, skip_reshape)


def get_attention_implementation(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Optional[Tensor] = None,
    attn_precision: Optional[torch.dtype] = None,
    skip_reshape: bool = False
) -> Tensor:
    """
    Factory function to get the most efficient attention implementation
    
    Args:
        q: Query tensor of shape [batch, seq_len, dim]
        k: Key tensor of shape [batch, seq_len, dim]  
        v: Value tensor of shape [batch, seq_len, dim]
        heads: Number of attention heads
        mask: Optional attention mask tensor
        attn_precision: Optional dtype for attention computation
        skip_reshape: Whether to skip reshape operations
    
    Returns:
        Tensor: Attention output tensor
    """
    device = q.device
    dtype = q.dtype
    impl_name, use_flash = memory_management.get_optimal_attention_implementation(device)
    
    # Process mask once
    batch_size = q.size(0)
    processed_mask = process_mask(mask, batch_size, heads, dtype, device) if mask is not None else None
    
    # Set computation precision
    computation_dtype = get_attention_precision(attn_precision)
    
    # Try Flash Attention 2 if available
    if use_flash and FLASH_ATTENTION_AVAILABLE:
        try:
            # Reshape inputs for Flash Attention if needed
            if not skip_reshape:
                q = q.reshape(-1, heads, q.shape[1], q.shape[2] // heads).contiguous()
                k = k.reshape(-1, heads, k.shape[1], k.shape[2] // heads).contiguous()
                v = v.reshape(-1, heads, v.shape[1], v.shape[2] // heads).contiguous()
            
            # Convert to computation dtype
            if computation_dtype != dtype:
                q = q.to(computation_dtype)
                k = k.to(computation_dtype)
                v = v.to(computation_dtype)
            
            # Flash Attention requires inputs in (B, S, H, D) format
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            output = flash_attn_func(
                q, k, v,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                window_size=(-1, -1) # No window size limit
            )
            
            # Reshape back to original format
            if not skip_reshape:
                output = output.transpose(1, 2).reshape(batch_size, -1, heads * (q.shape[-1]))
            
            # Convert back to original dtype if needed
            if computation_dtype != dtype:
                output = output.to(dtype)
                
            return output
            
        except Exception as e:
            warnings.warn(f"Flash Attention disabled due to error: {e}")
    
    # Try xFormers if available
    if impl_name == "xformers" and XFORMERS_AVAILABLE:
        try:
            return attention_xformers(q, k, v, heads, processed_mask, computation_dtype, skip_reshape)
        except Exception as e:
            warnings.warn(f"xFormers attention failed: {e}")
    
    # Use PyTorch's native attention if enabled
    if impl_name == "pytorch":
        try:
            return attention_pytorch(q, k, v, heads, processed_mask, computation_dtype, skip_reshape)
        except Exception as e:
            warnings.warn(f"PyTorch attention failed: {e}")
    
    # Fall back to memory-efficient custom implementations
    if impl_name == "sub_quad":
        return attention_sub_quad(q, k, v, heads, processed_mask, computation_dtype, skip_reshape)
    
    # Final fallback to split attention
    return attention_split(q, k, v, heads, processed_mask, computation_dtype, skip_reshape)

# Update the main attention dispatch
attention_function = get_attention_implementation


if memory_management.xformers_enabled_vae():
    print("Using xformers attention for VAE")
    attention_function_single_head_spatial = xformers_attention_single_head_spatial
elif memory_management.pytorch_attention_enabled():
    print("Using pytorch attention for VAE")
    attention_function_single_head_spatial = pytorch_attention_single_head_spatial
else:
    print("Using split attention for VAE")
    attention_function_single_head_spatial = normal_attention_single_head_spatial


class AttentionProcessorForge:
    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        hidden_states = attention_function(query, key, value, heads=attn.heads, mask=attention_mask)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# Expose module-level functions
__all__ = [
    'get_attn_precision',
    'efficient_dot_product_attention',
    'safe_import_xformers',
]
