import math
import torch
import torch as th
import torch.nn.functional as F

from typing import Optional, Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch import nn
from einops import rearrange, repeat
from backend.attention import attention_function

unet_initial_dtype = torch.float16
unet_initial_device = None


def checkpoint(f, args, parameters, enable=False):
    if enable:
        raise NotImplementedError('Gradient Checkpointing is not implemented.')
    return f(*args)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def apply_control(h, control, name):
    if control is not None and name in control and len(control[name]) > 0:
        ctrl = control[name].pop()
        if ctrl is not None:
            try:
                h += ctrl
            except:
                print("warning control could not be applied", h.shape, ctrl.shape)
    return h


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    # Consistent with Kohya to reduce differences between model training and inference.
    # Will be 0.005% slower than ComfyUI but Forge outweigh image quality than speed.

    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class TimestepBlock(nn.Module):
    pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, transformer_options={}, output_shape=None):
        block_inner_modifiers = transformer_options.get("block_inner_modifiers", [])

        for layer_index, layer in enumerate(self):
            for modifier in block_inner_modifiers:
                x = modifier(x, 'before', layer, layer_index, self, transformer_options)

            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, transformer_options)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, transformer_options)
                if "transformer_index" in transformer_options:
                    transformer_options["transformer_index"] += 1
            elif isinstance(layer, Upsample):
                x = layer(x, output_shape=output_shape)
            else:
                x = layer(x)

            for modifier in block_inner_modifiers:
                x = modifier(x, 'after', layer, layer_index, self, transformer_options)
        return x


class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, dtype=dtype, device=device)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., dtype=None, device=None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, dtype=dtype, device=device),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim, dtype=dtype, device=device)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, dtype=dtype, device=device)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., dtype=None, device=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

    def forward(self, x, context=None, value=None, mask=None, transformer_options={}):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        out = attention_function(q, k, v, self.heads, mask)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, ff_in=False,
                 inner_dim=None,
                 disable_self_attn=False, disable_temporal_crossattention=False, switch_temporal_ca_to_sa=False,
                 dtype=None, device=None):
        super().__init__()

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim, dtype=dtype, device=device)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device)

        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None, dtype=dtype,
                                    device=device)
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            context_dim_attn2 = None
            if not switch_temporal_ca_to_sa:
                context_dim_attn2 = context_dim

            self.attn2 = CrossAttention(query_dim=inner_dim, context_dim=context_dim_attn2,
                                        heads=n_heads, dim_head=d_head, dropout=dropout, dtype=dtype, device=device)
            self.norm2 = nn.LayerNorm(inner_dim, dtype=dtype, device=device)

        self.norm1 = nn.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = nn.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.checkpoint = checkpoint
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(self, x, context=None, transformer_options={}):
        return checkpoint(self._forward, (x, context, transformer_options), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, transformer_options={}):
        # Stolen from ComfyUI with some modifications

        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1, transformer_options=extra_options)

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if self.attn2 is not None:
            n = self.norm2(x)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2, transformer_options=extra_options)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        x_skip = 0

        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, dtype=None, device=None):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype,
                                 device=device)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0, dtype=dtype, device=device)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, dtype=dtype,
                                   device=device)
             for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = nn.Conv2d(inner_dim, in_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0, dtype=dtype, device=device)
        else:
            self.proj_out = nn.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=None, device=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding, dtype=dtype, device=device)

    def forward(self, x, output_shape=None):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
            if output_shape is not None:
                shape[1] = output_shape[3]
                shape[2] = output_shape[4]
        else:
            shape = [x.shape[2] * 2, x.shape[3] * 2]
            if output_shape is not None:
                shape[0] = output_shape[2]
                shape[1] = output_shape[3]

        x = F.interpolate(x, size=shape, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=None, device=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding, dtype=dtype, device=device
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            kernel_size=3,
            exchange_temb_dims=False,
            skip_t_emb=False,
            dtype=None,
            device=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        if self.skip_t_emb:
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels, dtype=dtype, device=device
                ),
            )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, dtype=dtype,
                    device=device)
            ,
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1, dtype=dtype, device=device)

    def forward(self, x, emb, transformer_options={}):
        return checkpoint(
            self._forward, (x, emb, transformer_options), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, transformer_options={}):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            if "groupnorm_wrapper" in transformer_options:
                in_norm, in_rest = in_rest[0], in_rest[1:]
                h = transformer_options["groupnorm_wrapper"](in_norm, x, transformer_options)
                h = in_rest(h)
            else:
                h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            if "groupnorm_wrapper" in transformer_options:
                in_norm = self.in_layers[0]
                h = transformer_options["groupnorm_wrapper"](in_norm, x, transformer_options)
                h = self.in_layers[1:](h)
            else:
                h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            if "groupnorm_wrapper" in transformer_options:
                h = transformer_options["groupnorm_wrapper"](out_norm, h, transformer_options)
            else:
                h = out_norm(h)
            if emb_out is not None:
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h *= (1 + scale)
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                h = h + emb_out
            if "groupnorm_wrapper" in transformer_options:
                h = transformer_options["groupnorm_wrapper"](self.out_layers[0], h, transformer_options)
                h = self.out_layers[1:](h)
            else:
                h = self.out_layers(h)
        return self.skip_connection(x) + h


class IntegratedUNet2DConditionModel(nn.Module, ConfigMixin):
    config_name = 'config.json'

    @register_to_config
    def __init__(self, sample_size: Optional[int] = None, in_channels: int = 4, out_channels: int = 4,
                 center_input_sample: bool = False, flip_sin_to_cos: bool = True, freq_shift: int = 0,
                 down_block_types: Tuple[str] = (
                         "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D",),
                 mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
                 up_block_types: Tuple[str] = (
                         "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                 only_cross_attention: Union[bool, Tuple[bool]] = False,
                 block_out_channels: Tuple[int] = (320, 640, 1280, 1280), layers_per_block: Union[int, Tuple[int]] = 2,
                 downsample_padding: int = 1, mid_block_scale_factor: float = 1, dropout: float = 0.0,
                 act_fn: str = "silu", norm_num_groups: Optional[int] = 32, norm_eps: float = 1e-5,
                 cross_attention_dim: Union[int, Tuple[int]] = 1280,
                 transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
                 reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
                 encoder_hid_dim: Optional[int] = None, encoder_hid_dim_type: Optional[str] = None,
                 attention_head_dim: Union[int, Tuple[int]] = 8,
                 num_attention_heads: Optional[Union[int, Tuple[int]]] = None, dual_cross_attention: bool = False,
                 use_linear_projection: bool = False, class_embed_type: Optional[str] = None,
                 addition_embed_type: Optional[str] = None, addition_time_embed_dim: Optional[int] = None,
                 num_class_embeds: Optional[int] = None, upcast_attention: bool = False,
                 resnet_time_scale_shift: str = "default", resnet_skip_time_act: bool = False,
                 resnet_out_scale_factor: float = 1.0, time_embedding_type: str = "positional",
                 time_embedding_dim: Optional[int] = None, time_embedding_act_fn: Optional[str] = None,
                 timestep_post_act: Optional[str] = None, time_cond_proj_dim: Optional[int] = None,
                 conv_in_kernel: int = 3, conv_out_kernel: int = 3,
                 projection_class_embeddings_input_dim: Optional[int] = None, attention_type: str = "default",
                 class_embeddings_concat: bool = False, mid_block_only_cross_attention: Optional[bool] = None,
                 cross_attention_norm: Optional[str] = None, addition_embed_type_num_heads: int = 64, *args, **kwargs):
        super().__init__()

        in_channels = in_channels
        out_channels = out_channels
        model_channels = block_out_channels[0]
        num_res_blocks = [layers_per_block] * len(block_out_channels)
        dropout = dropout
        channel_mult = [x // model_channels for x in block_out_channels]
        conv_resample = True
        dims = 2
        num_classes = None
        use_checkpoint = False
        adm_in_channels = None
        num_heads = -1
        num_head_channels = -1
        num_heads_upsample = -1
        use_scale_shift_norm = False
        resblock_updown = False
        use_spatial_transformer = True
        transformer_depth = []
        transformer_depth_output = []
        transformer_depth_middle = 1
        context_dim = cross_attention_dim
        disable_self_attentions: list = None
        num_attention_blocks: list = None
        disable_middle_self_attn = False
        use_linear_in_transformer = use_linear_projection

        for i, d in enumerate(down_block_types):
            if 'attn' in d.lower():
                current_transformer_depth = 1
                if isinstance(transformer_layers_per_block, list) and len(transformer_layers_per_block) > i:
                    current_transformer_depth = transformer_layers_per_block[i]
                transformer_depth += [current_transformer_depth] * 2
                transformer_depth_output += [current_transformer_depth] * 3
            else:
                transformer_depth += [0] * 2
                transformer_depth_output += [0] * 3

        if transformer_depth_output[-1] > 1:
            transformer_depth_middle = transformer_depth_output[-1]

        if isinstance(attention_head_dim, int):
            num_heads = attention_head_dim
        elif isinstance(attention_head_dim, list):
            num_head_channels = model_channels // attention_head_dim[0]
        else:
            raise ValueError('Wrong attention heads!')

        if isinstance(projection_class_embeddings_input_dim, int) and projection_class_embeddings_input_dim > 0:
            num_classes = 'sequential'
            adm_in_channels = projection_class_embeddings_input_dim

        dtype = unet_initial_dtype
        device = unet_initial_device

        if context_dim is not None:
            assert use_spatial_transformer

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("Bad num_res_blocks")
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)

        transformer_depth = transformer_depth[:]
        transformer_depth_output = transformer_depth_output[:]

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim, dtype=dtype, device=device)
            elif self.num_classes == "continuous":
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        nn.Linear(adm_in_channels, time_embed_dim, dtype=dtype, device=device),
                        nn.SiLU(),
                        nn.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device),
                    )
                )
            else:
                raise ValueError('Bad ADM')

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=dtype, device=device)
                )
            ]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=dtype,
                        device=device,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(SpatialTransformer(
                            ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim,
                            disable_self_attn=disabled_sa, use_checkpoint=use_checkpoint,
                            use_linear=use_linear_in_transformer, dtype=dtype, device=device)
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=dtype,
                            device=device,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, dtype=dtype, device=device,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        mid_block = [
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=None,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=dtype,
                device=device,
            )]
        if transformer_depth_middle >= 0:
            mid_block += [
                SpatialTransformer(
                    ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn, use_checkpoint=use_checkpoint,
                    use_linear=use_linear_in_transformer, dtype=dtype, device=device
                ),
                ResBlock(
                    channels=ch,
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    out_channels=None,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=dtype,
                    device=device,
                )]
        self.middle_block = TimestepEmbedSequential(*mid_block)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=dtype,
                        device=device,
                    )
                ]
                ch = model_channels * mult
                num_transformers = transformer_depth_output.pop()
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_checkpoint=use_checkpoint,
                                use_linear=use_linear_in_transformer, dtype=dtype, device=device
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=dtype,
                            device=device,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=dtype,
                                      device=device)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch, dtype=dtype, device=device),
            nn.SiLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1, dtype=dtype, device=device),
        )

    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})
        block_modifiers = transformer_options.get("block_modifiers", [])

        assert (y is not None) == (self.num_classes is not None)

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)

            for block_modifier in block_modifiers:
                h = block_modifier(h, 'before', transformer_options)

            h = module(h, emb, context, transformer_options)
            h = apply_control(h, control, 'input')

            for block_modifier in block_modifiers:
                h = block_modifier(h, 'after', transformer_options)

            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)

            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)

        for block_modifier in block_modifiers:
            h = block_modifier(h, 'before', transformer_options)

        h = self.middle_block(h, emb, context, transformer_options)
        h = apply_control(h, control, 'middle')

        for block_modifier in block_modifiers:
            h = block_modifier(h, 'after', transformer_options)

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')

            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)

            h = th.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None

            for block_modifier in block_modifiers:
                h = block_modifier(h, 'before', transformer_options)

            h = module(h, emb, context, transformer_options, output_shape)

            for block_modifier in block_modifiers:
                h = block_modifier(h, 'after', transformer_options)

        transformer_options["block"] = ("last", 0)

        for block_modifier in block_modifiers:
            h = block_modifier(h, 'before', transformer_options)

        if "groupnorm_wrapper" in transformer_options:
            out_norm, out_rest = self.out[0], self.out[1:]
            h = transformer_options["groupnorm_wrapper"](out_norm, h, transformer_options)
            h = out_rest(h)
        else:
            h = self.out(h)

        for block_modifier in block_modifiers:
            h = block_modifier(h, 'after', transformer_options)

        return h.type(x.dtype)
