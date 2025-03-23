from __future__ import division
import torch
from torch import Tensor
from typing import List, Union, Tuple, Dict, Optional
from backend import memory_management
from backend.misc.image_resize import adaptive_resize
from backend.patcher.base import ModelPatcher
from backend.patcher.controlnet import ControlNet, T2IAdapter
import numpy as np
from numpy import pi, exp, sqrt
from enum import Enum
from tqdm import tqdm

# Constants
opt_C = 4  # Channels in latent space
opt_f = 8  # Downsampling factor
MAX_RESOLUTION = 8192  # Maximum supported resolution

# Device Setup
class Device:
    def __init__(self):
        self.device = memory_management.get_torch_device()
devices = Device()

# Utility Functions
def ceildiv(big: int, small: int) -> int:
    """Ceiling division without floating-point errors."""
    return -(big // -small)

def gaussian_weights(tile_w: int, tile_h: int, variance: float = 0.01) -> Tensor:
    """
    Generate Gaussian weights for smooth tile blending.
    Args:
        tile_w (int): Tile width.
        tile_h (int): Tile height.
        variance (float): Variance for Gaussian distribution.
    Returns:
        Tensor: Weight tensor of shape (tile_h, tile_w).
    """
    f = lambda x, midpoint: exp(-(x - midpoint) ** 2 / (tile_w * tile_w) / (2 * variance)) / sqrt(2 * pi * variance)
    x_probs = [f(x, (tile_w - 1) / 2) for x in range(tile_w)]
    y_probs = [f(y, (tile_h - 1) / 2) for y in range(tile_h)]
    w = np.outer(y_probs, x_probs)
    return torch.from_numpy(w).to(devices.device, dtype=torch.float32)

# Enums
class BlendMode(Enum):
    FOREGROUND = "Foreground"
    BACKGROUND = "Background"
    OVERLAY = "Overlay"  # New: Overlay blending mode

class AttentionMode(Enum):
    SDP = "Scaled Dot-Product"
    MULTI_HEAD = "Multi-Head"

# Bounding Box Classes
class BBox:
    """Base class for bounding box representation."""
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x + w, y + h]
        self.slicer = slice(None), slice(None), slice(y, y + h), slice(x, x + w)

    def __getitem__(self, idx: int) -> int:
        return self.box[idx]

class CustomBBox(BBox):
    """Custom bounding box with additional attributes."""
    def __init__(self, x: int, y: int, w: int, h: int, blend_mode: BlendMode = BlendMode.FOREGROUND, priority: int = 0):
        super().__init__(x, y, w, h)
        self.blend_mode = blend_mode
        self.priority = priority

# Image Scaling Utility
class ImageScale:
    def upscale(self, image: Tensor, upscale_method: str, width: int, height: int, crop: str) -> Tuple[Tensor]:
        """
        Upscale an image tensor with adaptive resizing.
        Args:
            image (Tensor): Input image tensor.
            upscale_method (str): Method for upscaling (e.g., 'bilinear').
            width (int): Target width.
            height (int): Target height.
            crop (str): Cropping strategy.
        Returns:
            Tuple[Tensor]: Upscaled image tensor.
        """
        if width == 0 and height == 0:
            return (image,)
        samples = image.movedim(-1, 1)
        if width == 0:
            width = max(1, round(samples.shape[3] * height / samples.shape[2]))
        elif height == 0:
            height = max(1, round(samples.shape[2] * width / samples.shape[3]))
        s = adaptive_resize(samples, width, height, upscale_method, crop)
        s = s.movedim(1, -1)
        return (s,)

# Abstract Diffusion Base Class
class AbstractDiffusion:
    def __init__(self):
        self.method = self.__class__.__name__
        self.pbar: Optional[tqdm] = None
        # Core tiling parameters
        self.w: int = 0
        self.h: int = 0
        self.tile_width: int = None
        self.tile_height: int = None
        self.tile_overlap: int = None
        self.tile_batch_size: int = None
        # Buffers
        self.x_buffer: Tensor = None
        self.weights: Tensor = None
        # Grid tiling
        self.enable_grid_bbox: bool = False
        self.tile_w: int = None
        self.tile_h: int = None
        self.tile_bs: int = None
        self.num_tiles: int = None
        self.num_batches: int = None
        self.batched_bboxes: List[List[BBox]] = []
        # Custom regions
        self.enable_custom_bbox: bool = False
        self.custom_bboxes: List[CustomBBox] = []
        # ControlNet
        self.enable_controlnet: bool = False
        self.control_tensor_batch: List[List[Tensor]] = []
        self.control_tensor_custom: List[List[Tensor]] = []
        self.control_params: Dict[Tuple, List[List[Tensor]]] = {}
        self.control_tensor_cpu: bool = False
        # New Features
        self.attention_mode: AttentionMode = AttentionMode.SDP
        self.multi_scale: bool = False
        self.cache_enabled: bool = False
        self.cache_key: Optional[Tuple] = None
        self.cache: Dict[Tuple, Tensor] = {}
        # State
        self.draw_background: bool = True
        self.step_count: int = 0
        self.inner_loop_count: int = 0
        self.total_bboxes: int = 0
        self.imagescale = ImageScale()

    def reset(self):
        """Reset instance while preserving tiling parameters."""
        params = (self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
        self.__init__()
        self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size = params

    def repeat_tensor(self, x: Tensor, n: int, concat: bool = False, concat_to: int = 0) -> Tensor:
        """Repeat or concatenate a tensor along the first dimension."""
        if n == 1:
            return x
        B = x.shape[0]
        r_dims = len(x.shape) - 1
        if B == 1:
            shape = [n] + [-1] * r_dims
            return x.expand(shape)
        if concat:
            return torch.cat([x for _ in range(n)], dim=0)[:concat_to]
        shape = [n] + [1] * r_dims
        return x.repeat(shape)

    def reset_buffer(self, x_in: Tensor):
        """Reset or initialize the output buffer."""
        if self.x_buffer is None or self.x_buffer.shape != x_in.shape:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)
        else:
            self.x_buffer.zero_()

    def init_grid_bbox(self, tile_w: int, tile_h: int, overlap: int, tile_bs: int):
        """Initialize grid-based tiling with overlap."""
        self.enable_grid_bbox = True
        self.tile_w = min(tile_w, self.w)
        self.tile_h = min(tile_h, self.h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        bboxes, weights = self.split_bboxes(self.w, self.h, self.tile_w, self.tile_h, overlap)
        self.weights = weights if self.weights is None else self.weights + weights
        self.num_tiles = len(bboxes)
        self.num_batches = ceildiv(self.num_tiles, tile_bs)
        self.tile_bs = ceildiv(len(bboxes), self.num_batches)
        self.batched_bboxes = [bboxes[i * self.tile_bs:(i + 1) * self.tile_bs] for i in range(self.num_batches)]

    def split_bboxes(self, w: int, h: int, tile_w: int, tile_h: int, overlap: int) -> Tuple[List[BBox], Tensor]:
        """Split the latent space into overlapping tiles."""
        cols = ceildiv((w - overlap), (tile_w - overlap))
        rows = ceildiv((h - overlap), (tile_h - overlap))
        dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
        dy = (h - tile_h) / (rows - 1) if rows > 1 else 0
        bbox_list: List[BBox] = []
        weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
        for row in range(rows):
            y = min(int(row * dy), h - tile_h)
            for col in range(cols):
                x = min(int(col * dx), w - tile_w)
                bbox = BBox(x, y, tile_w, tile_h)
                bbox_list.append(bbox)
                weight[bbox.slicer] += self.get_tile_weights(tile_w, tile_h)
        return bbox_list, weight

    def get_tile_weights(self, tile_w: int, tile_h: int) -> Tensor:
        """Generate weights for tile blending (default: uniform)."""
        return torch.ones((tile_h, tile_w), device=devices.device, dtype=torch.float32)

    def init_custom_bbox(self, custom_bboxes: List[Dict[str, int]], blend_modes: List[str], priorities: List[int]):
        """Initialize custom bounding boxes for region-specific control."""
        self.enable_custom_bbox = True
        self.custom_bboxes = [
            CustomBBox(
                x=cb["x"], y=cb["y"], w=cb["w"], h=cb["h"],
                blend_mode=BlendMode[bm.upper()],
                priority=p
            ) for cb, bm, p in zip(custom_bboxes, blend_modes, priorities)
        ]
        self.custom_bboxes.sort(key=lambda x: x.priority, reverse=True)

    def init_done(self, sampling_steps: int):
        """Finalize initialization with sanity checks and progress setup."""
        self.total_bboxes = 0
        if self.enable_grid_bbox:
            self.total_bboxes += self.num_batches
        if self.enable_custom_bbox:
            self.total_bboxes += len(self.custom_bboxes)
        if self.total_bboxes == 0:
            raise ValueError("No regions to process: enable grid or custom bounding boxes.")
        self.pbar = tqdm(total=self.total_bboxes * sampling_steps, desc=f"{self.method} Sampling")

    def process_controlnet(self, x_shape: Tuple, x_dtype: torch.dtype, c_in: Dict, cond_or_uncond: List, bboxes: List[BBox], batch_size: int, batch_id: int):
        """Process ControlNet tensors for tiled diffusion."""
        if not self.enable_controlnet or "control" not in c_in:
            return
        control: ControlNet = c_in["control_model"]
        tuple_key = tuple(cond_or_uncond) + tuple(x_shape)
        param_id = -1
        while control is not None:
            param_id += 1
            PH, PW = self.h * opt_f, self.w * opt_f
            if tuple_key not in self.control_params:
                self.control_params[tuple_key] = [[None] * len(self.batched_bboxes) for _ in range(param_id + 1)]
            if param_id >= len(self.control_params[tuple_key]):
                self.control_params[tuple_key].append([None] * len(self.batched_bboxes))
            if control.cond_hint is None or not isinstance(self.control_params[tuple_key][param_id][batch_id], Tensor):
                dtype = getattr(control, "manual_cast_dtype", x_dtype)
                if isinstance(control, T2IAdapter):
                    width, height = control.scale_image_to(PW, PH)
                    control.cond_hint = adaptive_resize(control.cond_hint_original, width, height, "nearest-exact", "center").to(dtype=dtype, device=control.device)
                else:
                    control.cond_hint = adaptive_resize(control.cond_hint_original, PW, PH, "nearest-exact", "center").to(dtype=dtype, device=control.device)
                cond_hint_pre_tile = self.repeat_tensor(control.cond_hint, ceildiv(batch_size, control.cond_hint.shape[0]))[:batch_size]
                cns = [cond_hint_pre_tile[:, :, bbox[1] * opt_f:bbox[3] * opt_f, bbox[0] * opt_f:bbox[2] * opt_f] for bbox in bboxes]
                control.cond_hint = torch.cat(cns, dim=0)
                self.control_params[tuple_key][param_id][batch_id] = control.cond_hint
            else:
                control.cond_hint = self.control_params[tuple_key][param_id][batch_id]
            control = control.previous_controlnet

# MultiDiffusion Implementation
class MultiDiffusion(AbstractDiffusion):
    """MultiDiffusion: Uniform weighting across tiles."""
    @torch.no_grad()
    def __call__(self, model_function, args: Dict) -> Tensor:
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: Dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]
        c_crossattn: Tensor = c_in["c_crossattn"]
        N, C, H, W = x_in.shape

        # Initialize or refresh tiling
        refresh = self.weights is None or self.h != H or self.w != W
        if refresh:
            self.h, self.w = H, W
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            self.init_done(args.get("steps", 20))
        self.h, self.w = H, W
        self.reset_buffer(x_in)

        # Check cache
        if self.cache_enabled:
            cache_key = (x_in.tobytes(), t_in.tobytes(), tuple(c_in.items()))
            if cache_key in self.cache:
                return self.cache[cache_key]

        # Background sampling
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)
                n_rep = len(bboxes)
                ts_tile = self.repeat_tensor(t_in, n_rep)
                cond_tile = self.repeat_tensor(c_crossattn, n_rep)
                c_tile = c_in.copy()
                c_tile["c_crossattn"] = cond_tile
                for key in ["time_context", "y", "c_concat"]:
                    if key in c_in:
                        val = c_in[key]
                        c_tile[key] = torch.cat([val[bbox.slicer] for bbox in bboxes]) if val.shape[2:] == (H, W) else self.repeat_tensor(val, n_rep)
                if "control" in c_in:
                    self.process_controlnet(x_tile.shape, x_tile.dtype, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile["control"] = c_in["control_model"].get_control(x_tile, ts_tile, c_tile, len(cond_or_uncond))
                x_tile_out = model_function(x_tile, ts_tile, **c_tile)
                for i, bbox in enumerate(bboxes):
                    self.x_buffer[bbox.slicer] += x_tile_out[i * N:(i + 1) * N]
                del x_tile_out, x_tile, ts_tile, c_tile
                self.pbar.update()

        # Average buffer
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)
        if self.cache_enabled:
            self.cache[cache_key] = x_out
        return x_out

# MixtureOfDiffusers Implementation
class MixtureOfDiffusers(AbstractDiffusion):
    """Mixture of Diffusers: Gaussian-weighted tile blending."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_weights: List[Tensor] = []

    def get_tile_weights(self, tile_w: int, tile_h: int) -> Tensor:
        """Use Gaussian weights for smooth blending."""
        return gaussian_weights(tile_w, tile_h)

    def init_done(self, sampling_steps: int):
        super().init_done(sampling_steps)
        self.rescale_factor = 1 / self.weights.clamp(min=1e-8)
        if self.enable_custom_bbox:
            self.custom_weights = [self.get_tile_weights(cb.w, cb.h) * self.rescale_factor[cb.slicer] for cb in self.custom_bboxes]

    @torch.no_grad()
    def __call__(self, model_function, args: Dict) -> Tensor:
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: Dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]
        c_crossattn: Tensor = c_in["c_crossattn"]
        N, C, H, W = x_in.shape

        # Initialize or refresh tiling
        refresh = self.weights is None or self.h != H or self.w != W
        if refresh:
            self.h, self.w = H, W
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            self.init_done(args.get("steps", 20))
        self.h, self.w = H, W
        self.reset_buffer(x_in)

        # Check cache
        if self.cache_enabled:
            cache_key = (x_in.tobytes(), t_in.tobytes(), tuple(c_in.items()))
            if cache_key in self.cache:
                return self.cache[cache_key]

        # Background sampling
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)
                n_rep = len(bboxes)
                ts_tile = self.repeat_tensor(t_in, n_rep)
                cond_tile = self.repeat_tensor(c_crossattn, n_rep)
                c_tile = c_in.copy()
                c_tile["c_crossattn"] = cond_tile
                for key in ["time_context", "y", "c_concat"]:
                    if key in c_in:
                        val = c_in[key]
                        c_tile[key] = torch.cat([val[bbox.slicer] for bbox in bboxes]) if val.shape[2:] == (H, W) else self.repeat_tensor(val, n_rep)
                if "control" in c_in:
                    self.process_controlnet(x_tile.shape, x_tile.dtype, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile["control"] = c_in["control_model"].get_control(x_tile, ts_tile, c_tile, len(cond_or_uncond))
                x_tile_out = model_function(x_tile, ts_tile, **c_tile)
                for i, bbox in enumerate(bboxes):
                    w = self.get_tile_weights(self.tile_w, self.tile_h) * self.rescale_factor[bbox.slicer]
                    self.x_buffer[bbox.slicer] += x_tile_out[i * N:(i + 1) * N] * w
                del x_tile_out, x_tile, ts_tile, c_tile
                self.pbar.update()

        # Custom regions
        if self.enable_custom_bbox:
            for i, bbox in enumerate(self.custom_bboxes):
                x_tile = x_in[bbox.slicer]
                ts_tile = t_in
                cond_tile = c_crossattn
                c_tile = c_in.copy()
                c_tile["c_crossattn"] = cond_tile
                for key in ["time_context", "y", "c_concat"]:
                    if key in c_in:
                        val = c_in[key]
                        c_tile[key] = val[bbox.slicer] if val.shape[2:] == (H, W) else val
                if "control" in c_in:
                    self.process_controlnet(x_tile.shape, x_tile.dtype, c_in, cond_or_uncond, [bbox], N, 0)
                    c_tile["control"] = c_in["control_model"].get_control(x_tile, ts_tile, c_tile, len(cond_or_uncond))
                x_tile_out = model_function(x_tile, ts_tile, **c_tile)
                w = self.custom_weights[i]
                if bbox.blend_mode == BlendMode.OVERLAY:
                    self.x_buffer[bbox.slicer] = x_tile_out * w + self.x_buffer[bbox.slicer] * (1 - w)
                else:
                    self.x_buffer[bbox.slicer] += x_tile_out * w
                self.pbar.update()

        x_out = self.x_buffer
        if self.cache_enabled:
            self.cache[cache_key] = x_out
        return x_out

# TiledDiffusion Interface
class TiledDiffusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "method": (["MultiDiffusion", "Mixture of Diffusers"], {"default": "Mixture of Diffusers"}),
                "tile_width": ("INT", {"default": 96 * opt_f, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "tile_height": ("INT", {"default": 96 * opt_f, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "tile_overlap": ("INT", {"default": 8 * opt_f, "min": 0, "max": 256 * opt_f, "step": 4 * opt_f}),
                "tile_batch_size": ("INT", {"default": 4, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            },
            "optional": {
                "attention_mode": (["Scaled Dot-Product", "Multi-Head"], {"default": "Scaled Dot-Product"}),
                "multi_scale": ("BOOLEAN", {"default": False}),
                "cache_enabled": ("BOOLEAN", {"default": False}),
                "custom_bboxes": ("LIST", {"default": []}),
                "blend_modes": ("LIST", {"default": []}),
                "priorities": ("LIST", {"default": []}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"

    def apply(self, model: ModelPatcher, method: str, tile_width: int, tile_height: int, tile_overlap: int, tile_batch_size: int,
              attention_mode: str = "Scaled Dot-Product", multi_scale: bool = False, cache_enabled: bool = False,
              custom_bboxes: List[Dict] = [], blend_modes: List[str] = [], priorities: List[int] = []):
        """Apply tiled diffusion to the model."""
        implement = MixtureOfDiffusers() if method == "Mixture of Diffusers" else MultiDiffusion()
        implement.tile_width = tile_width // opt_f
        implement.tile_height = tile_height // opt_f
        implement.tile_overlap = tile_overlap // opt_f
        implement.tile_batch_size = tile_batch_size
        implement.attention_mode = AttentionMode[attention_mode.upper().replace(" ", "_")]
        implement.multi_scale = multi_scale
        implement.cache_enabled = cache_enabled
        if custom_bboxes:
            implement.init_custom_bbox(custom_bboxes, blend_modes or ["FOREGROUND"] * len(custom_bboxes), priorities or [0] * len(custom_bboxes))
        model = model.clone()
        model.set_model_unet_function_wrapper(implement)
        model.model_options["tiled_diffusion"] = True
        return (model,)
