import math
import torch
import einops

from backend.args import args
from backend import memory_management
from backend.misc.sub_quadratic_attention import efficient_dot_product_attention


BROKEN_XFORMERS = False
if memory_management.xformers_enabled():
    import xformers
    import xformers.ops

    try:
        x_vers = xformers.__version__
        BROKEN_XFORMERS = x_vers.startswith("0.0.2") and not x_vers.startswith("0.0.20")
    except:
        pass

if memory_management.sage_attention_enabled():
    try:
        from sageattention import sageattn
    except ModuleNotFoundError:
        print(f"\n\nTo use the `--use-sage-attention` feature, the `sageattention` package must be installed first.\ncommand:\n\t{sys.executable} -m pip install sageattention")
        exit(-1)

if memory_management.flash_attention_enabled():
    try:
        from flash_attn import flash_attn_func
    except ModuleNotFoundError:
        print(f"\n\nTo use the `--use-flash-attention` feature, the `flash-attn` package must be installed first.\ncommand:\n\t{sys.executable} -m pip install flash-attn")
        exit(-1)

import backend.operations
ops = backend.operations.ForgeOperations

FORCE_UPCAST_ATTENTION_DTYPE = memory_management.force_upcast_attention_dtype()


def get_attn_precision(attn_precision=torch.float32):
    if args.disable_attention_upcast:
        return None
    if FORCE_UPCAST_ATTENTION_DTYPE is not None:
        return FORCE_UPCAST_ATTENTION_DTYPE
    return attn_precision


def exists(val):
    return val is not None


def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    attn_precision = get_attn_precision(attn_precision)

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
    attn_precision = get_attn_precision(attn_precision)

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
    attn_precision = get_attn_precision(attn_precision)

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
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    out = (
        out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    )
    return out

def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
    # sageattn doesn't work with sd1.5
    if q.shape[-1] // heads not in [64, 96, 128]:
        if memory_management.flash_attention_enabled():
            return attention_flash(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape)
        elif memory_management.xformers_enabled():
            return attention_xformers(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape)
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape)
    if skip_reshape:
        b, _, _, dim_head = q.shape
        tensor_layout="HND"
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head),
            (q, k, v),
        )
        tensor_layout="NHD"

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    out = sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
    if tensor_layout == "HND":
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
    else:
        if skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(b, -1, heads * dim_head)
    return out

try:
    @torch.library.custom_op("flash_attention::flash_attn", mutates_args=())
    def flash_attn_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    dropout_p: float = 0.0, causal: bool = False) -> torch.Tensor:
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)


    @flash_attn_wrapper.register_fake
    def flash_attn_fake(q, k, v, dropout_p=0.0, causal=False):
        # Output shape is the same as q
        return q.new_empty(q.shape)
except AttributeError as error:
    FLASH_ATTN_ERROR = error

    def flash_attn_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    dropout_p: float = 0.0, causal: bool = False) -> torch.Tensor:
        assert False, f"Could not define flash_attn_wrapper: {FLASH_ATTN_ERROR}"


def attention_flash(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    try:
        assert mask is None
        out = flash_attn_wrapper(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            causal=False,
        ).transpose(1, 2)
    except Exception as e:
        print(f"Flash Attention failed, using default SDPA: {e}")
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    if not skip_output_reshape:
        out = (
            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        )
    return out

def slice_attention_single_head_spatial(q, k, v):
    r1 = torch.zeros_like(k, device=q.device)
    scale = (int(q.shape[-1]) ** (-0.5))

    mem_free_total = memory_management.get_free_memory(q.device)

    gb = 1024 ** 3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

    while True:
        try:
            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                s1 = torch.bmm(q[:, i:end], k) * scale

                s2 = torch.nn.functional.softmax(s1, dim=2).permute(0, 2, 1)
                del s1

                r1[:, :, i:end] = torch.bmm(v, s2)
                del s2
            break
        except memory_management.OOM_EXCEPTION as e:
            memory_management.soft_empty_cache(True)
            steps *= 2
            if steps > 128:
                raise e
            print("out of memory error, increasing steps and trying again {}".format(steps))

    return r1


def normal_attention_single_head_spatial(q, k, v):
    # compute attention
    b, c, h, w = q.shape

    q = q.reshape(b, c, h * w)
    q = q.permute(0, 2, 1)  # b,hw,c
    k = k.reshape(b, c, h * w)  # b,c,hw
    v = v.reshape(b, c, h * w)

    r1 = slice_attention_single_head_spatial(q, k, v)
    h_ = r1.reshape(b, c, h, w)
    del r1
    return h_


def xformers_attention_single_head_spatial(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )

    try:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        out = out.transpose(1, 2).reshape(B, C, H, W)
    except NotImplementedError as e:
        out = slice_attention_single_head_spatial(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2),
                                                  v.view(B, -1, C).transpose(1, 2)).reshape(B, C, H, W)
    return out


def pytorch_attention_single_head_spatial(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )

    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).reshape(B, C, H, W)
    except memory_management.OOM_EXCEPTION as e:
        print("scaled_dot_product_attention OOMed: switched to slice attention")
        out = slice_attention_single_head_spatial(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2),
                                                  v.view(B, -1, C).transpose(1, 2)).reshape(B, C, H, W)
    return out


if memory_management.sage_attention_enabled():
    print("Using sage attention")
    attention_function = attention_sage
elif memory_management.flash_attention_enabled():
    print("Using Flash Attention")
    attention_function = attention_flash
elif memory_management.xformers_enabled():
    print("Using xformers cross attention")
    attention_function = attention_xformers
elif memory_management.pytorch_attention_enabled():
    print("Using pytorch cross attention")
    attention_function = attention_pytorch
elif args.attention_split:
    print("Using split optimization for cross attention")
    attention_function = attention_split
else:
    print("Using sub quadratic optimization for cross attention")
    attention_function = attention_sub_quad

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
