import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Set manual seeds for noise
# rand(n)_like. but with generator support
def gen_like(f, input, generator=None):
    return f(input.size(), generator=generator).to(input)

'''
    The following snippet is utilized from https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/
'''
def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return mn#torch.clamp(mn, min=-1)
    
def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return mx#torch.clamp(mx, max=1)
def contrast_adaptive_sharpening(image, amount):
    img = F.pad(image, pad=(1, 1, 1, 1))
    absmean = torch.abs(image.mean())

    a = img[..., :-2, :-2]
    b = img[..., :-2, 1:-1]
    c = img[..., :-2, 2:]
    d = img[..., 1:-1, :-2]
    e = img[..., 1:-1, 1:-1]
    f = img[..., 1:-1, 2:]
    g = img[..., 2:, :-2]
    h = img[..., 2:, 1:-1]
    i = img[..., 2:, 2:]
    
    # Computing contrast
    cross = (b, d, e, f, h)
    mn = min_(cross)
    mx = max_(cross)
    
    diag = (a, c, g, i)
    mn2 = min_(diag)
    mx2 = max_(diag)
    mx = mx + mx2
    mn = mn + mn2
    
    # Computing local weight
    inv_mx = torch.reciprocal(mx)
    amp = inv_mx * torch.minimum(mn, (2 - mx))

    # scaling
    amp = torch.copysign(torch.sqrt(torch.abs(amp)), amp)
    w = - amp * (amount * (1/5 - 1/8) + 1/8)
    div = torch.reciprocal(1 + 4*w).clamp(-10, 10)
    
    output = ((b + d + f + h)*w + e) * div
    output = torch.nan_to_num(output)

    return (output.to(image.device))

'''
    The following gaussian functions were utilized from the Fooocus UI, many thanks to github.com/Illyasviel !
'''
def gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    return kernel / np.sum(kernel)


class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2  # Ensure output size matches input size
        self.register_buffer('kernel', torch.tensor(gaussian_kernel(kernel_size, sigma), dtype=torch.float32))
        self.kernel = self.kernel.view(1, 1, kernel_size, kernel_size)
        self.kernel = self.kernel.expand(self.channels, -1, -1, -1)  # Repeat the kernel for each input channel

    def forward(self, x):
        x = F.conv2d(x, self.kernel.to(x), padding=self.padding, groups=self.channels)
        return x

gaussian_filter_2d = GaussianBlur(4, 7, 0.8)

'''
    As of August 18th (on Fooocus' GitHub), the gaussian functions were replaced by an anisotropic function for better stability.
'''
Tensor = torch.Tensor
Device = torch.DeviceObjType
Dtype = torch.Type
pad = torch.nn.functional.pad


def _compute_zero_padding(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(kernel_size) == 2, '2D Kernel size should have a length of 2.'
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)
    return ky, kx


def gaussian(
    window_size: int, sigma: Tensor | float, *, device: Device | None = None, dtype: Dtype | None = None
) -> Tensor:

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def get_gaussian_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> Tensor:

    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_kernel2d(
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> Tensor:

    sigma = torch.Tensor([[sigma, sigma]]).to(device=device, dtype=dtype)

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def _bilateral_blur(
    input: Tensor,
    guidance: Tensor | None,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:

    if isinstance(sigma_color, Tensor):
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(-1, 1, 1, 1, 1)

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = pad(guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
        unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only acceps l1 or l2")
    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, device=input.device, dtype=input.dtype)
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


def bilateral_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int = (13, 13),
    sigma_color: float | Tensor = 3.0,
    sigma_space: tuple[float, float] | Tensor = 3.0,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:
    return _bilateral_blur(input, None, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


def joint_bilateral_blur(
    input: Tensor,
    guidance: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:
    return _bilateral_blur(input, guidance, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


class _BilateralBlur(torch.nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma_color: float | Tensor,
        sigma_space: tuple[float, float] | Tensor,
        border_type: str = 'reflect',
        color_distance_type: str = "l1",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type
        self.color_distance_type = color_distance_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space}, "
            f"border_type={self.border_type}, "
            f"color_distance_type={self.color_distance_type})"
        )


class BilateralBlur(_BilateralBlur):
    def forward(self, input: Tensor) -> Tensor:
        return bilateral_blur(
            input, self.kernel_size, self.sigma_color, self.sigma_space, self.border_type, self.color_distance_type
        )


class JointBilateralBlur(_BilateralBlur):
    def forward(self, input: Tensor, guidance: Tensor) -> Tensor:
        return joint_bilateral_blur(
            input,
            guidance,
            self.kernel_size,
            self.sigma_color,
            self.sigma_space,
            self.border_type,
            self.color_distance_type,
        )


# Below is perlin noise from https://github.com/tasptz/pytorch-perlin-noise/blob/main/perlin_noise/perlin_noise.py
from torch import Generator, Tensor, lerp
from torch.nn.functional import unfold
from typing import Callable, Tuple
from math import pi

def get_positions(block_shape: Tuple[int, int]) -> Tensor:
    """
    Generate position tensor.

    Arguments:
        block_shape -- (height, width) of position tensor

    Returns:
        position vector shaped (1, height, width, 1, 1, 2)
    """
    bh, bw = block_shape
    positions = torch.stack(
        torch.meshgrid(
            [(torch.arange(b) + 0.5) / b for b in (bw, bh)],
            indexing="xy",
        ),
        -1,
    ).view(1, bh, bw, 1, 1, 2)
    return positions


def unfold_grid(vectors: Tensor) -> Tensor:
    """
    Unfold vector grid to batched vectors.

    Arguments:
        vectors -- grid vectors

    Returns:
        batched grid vectors
    """
    batch_size, _, gpy, gpx = vectors.shape
    return (
        unfold(vectors, (2, 2))
        .view(batch_size, 2, 4, -1)
        .permute(0, 2, 3, 1)
        .view(batch_size, 4, gpy - 1, gpx - 1, 2)
    )


def smooth_step(t: Tensor) -> Tensor:
    """
    Smooth step function [0, 1] -> [0, 1].

    Arguments:
        t -- input values (any shape)

    Returns:
        output values (same shape as input values)
    """
    return t * t * (3.0 - 2.0 * t)


def perlin_noise_tensor(
    vectors: Tensor, positions: Tensor, step: Callable = None
) -> Tensor:
    """
    Generate perlin noise from batched vectors and positions.

    Arguments:
        vectors -- batched grid vectors shaped (batch_size, 4, grid_height, grid_width, 2)
        positions -- batched grid positions shaped (batch_size or 1, block_height, block_width, grid_height or 1, grid_width or 1, 2)

    Keyword Arguments:
        step -- smooth step function [0, 1] -> [0, 1] (default: `smooth_step`)

    Raises:
        Exception: if position and vector shapes do not match

    Returns:
        (batch_size, block_height * grid_height, block_width * grid_width)
    """
    if step is None:
        step = smooth_step

    batch_size = vectors.shape[0]
    # grid height, grid width
    gh, gw = vectors.shape[2:4]
    # block height, block width
    bh, bw = positions.shape[1:3]

    for i in range(2):
        if positions.shape[i + 3] not in (1, vectors.shape[i + 2]):
            raise Exception(
                f"Blocks shapes do not match: vectors ({vectors.shape[1]}, {vectors.shape[2]}), positions {gh}, {gw})"
            )

    if positions.shape[0] not in (1, batch_size):
        raise Exception(
            f"Batch sizes do not match: vectors ({vectors.shape[0]}), positions ({positions.shape[0]})"
        )

    vectors = vectors.view(batch_size, 4, 1, gh * gw, 2)
    positions = positions.view(positions.shape[0], bh * bw, -1, 2)

    step_x = step(positions[..., 0])
    step_y = step(positions[..., 1])

    row0 = lerp(
        (vectors[:, 0] * positions).sum(dim=-1),
        (vectors[:, 1] * (positions - positions.new_tensor((1, 0)))).sum(dim=-1),
        step_x,
    )
    row1 = lerp(
        (vectors[:, 2] * (positions - positions.new_tensor((0, 1)))).sum(dim=-1),
        (vectors[:, 3] * (positions - positions.new_tensor((1, 1)))).sum(dim=-1),
        step_x,
    )
    noise = lerp(row0, row1, step_y)
    return (
        noise.view(
            batch_size,
            bh,
            bw,
            gh,
            gw,
        )
        .permute(0, 3, 1, 4, 2)
        .reshape(batch_size, gh * bh, gw * bw)
    )


def perlin_noise(
    grid_shape: Tuple[int, int],
    out_shape: Tuple[int, int],
    batch_size: int = 1,
    generator: Generator = None,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate perlin noise with given shape. `*args` and `**kwargs` are forwarded to `Tensor` creation.

    Arguments:
        grid_shape -- Shape of grid (height, width).
        out_shape -- Shape of output noise image (height, width).

    Keyword Arguments:
        batch_size -- (default: {1})
        generator -- random generator used for grid vectors (default: {None})

    Raises:
        Exception: if grid and out shapes do not match

    Returns:
        Noise image shaped (batch_size, height, width)
    """
    # grid height and width
    gh, gw = grid_shape
    # output height and width
    oh, ow = out_shape
    # block height and width
    bh, bw = oh // gh, ow // gw

    if oh != bh * gh:
        raise Exception(f"Output height {oh} must be divisible by grid height {gh}")
    if ow != bw * gw != 0:
        raise Exception(f"Output width {ow} must be divisible by grid width {gw}")

    angle = torch.empty(
        [batch_size] + [s + 1 for s in grid_shape], *args, **kwargs
    ).uniform_(to=2.0 * pi, generator=generator)
    # random vectors on grid points
    vectors = unfold_grid(torch.stack((torch.cos(angle), torch.sin(angle)), dim=1))
    # positions inside grid cells [0, 1)
    positions = get_positions((bh, bw)).to(vectors)
    return perlin_noise_tensor(vectors, positions).squeeze(0)

def generate_1f_noise(tensor, alpha, k, generator=None):
    """Generate 1/f noise for a given tensor.

    Args:
        tensor: The tensor to add noise to.
        alpha: The parameter that determines the slope of the spectrum.
        k: A constant.

    Returns:
        A tensor with the same shape as `tensor` containing 1/f noise.
    """
    fft = torch.fft.fft2(tensor)
    freq = torch.arange(1, len(fft) + 1, dtype=torch.float)
    spectral_density = k / freq**alpha
    noise = torch.randn(tensor.shape, generator=generator) * spectral_density
    return noise

def green_noise(width, height, generator=None):
    noise = torch.randn(width, height, generator=generator)
    scale = 1.0 / (width * height)
    fy = torch.fft.fftfreq(width)[:, None] ** 2
    fx = torch.fft.fftfreq(height) ** 2
    f = fy + fx
    power = torch.sqrt(f)
    power[0, 0] = 1
    noise = torch.fft.ifft2(torch.fft.fft2(noise) / torch.sqrt(power))
    noise *= scale / noise.std()
    return torch.real(noise)

# Algorithm from https://github.com/v0xie/sd-webui-cads/
def add_cads_noise(y, timestep, cads_schedule_start, cads_schedule_end, cads_noise_scale, cads_rescale_factor, cads_rescale=False):
    timestep_as_float = (timestep / 999.0)[:, None, None, None].clone()[0].item()
    gamma = 0.0
    if timestep_as_float < cads_schedule_start:
        gamma = 1.0
    elif timestep_as_float > cads_schedule_end:
        gamma = 0.0
    else: 
        gamma = (cads_schedule_end - timestep_as_float) / (cads_schedule_end - cads_schedule_start)

    y_mean, y_std = torch.mean(y), torch.std(y)
    y = np.sqrt(gamma) * y + cads_noise_scale * np.sqrt(1 - gamma) * torch.randn_like(y)

    if cads_rescale:
        y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
        if not torch.isnan(y_scaled).any():
            y = cads_rescale_factor * y_scaled + (1 - cads_rescale_factor) * y
        else:
            print("Encountered NaN in cads rescaling. Skipping rescaling.")
    return y

# Algorithm from https://github.com/v0xie/sd-webui-cads/
def add_cads_custom_noise(y, noise, timestep, cads_schedule_start, cads_schedule_end, cads_noise_scale, cads_rescale_factor, cads_rescale=False):
    timestep_as_float = (timestep / 999.0)[:, None, None, None].clone()[0].item()
    gamma = 0.0
    if timestep_as_float < cads_schedule_start:
        gamma = 1.0
    elif timestep_as_float > cads_schedule_end:
        gamma = 0.0
    else: 
        gamma = (cads_schedule_end - timestep_as_float) / (cads_schedule_end - cads_schedule_start)

    y_mean, y_std = torch.mean(y), torch.std(y)
    y = np.sqrt(gamma) * y + cads_noise_scale * np.sqrt(1 - gamma) * noise#.sub_(noise.mean()).div_(noise.std())

    if cads_rescale:
        y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
        if not torch.isnan(y_scaled).any():
            y = cads_rescale_factor * y_scaled + (1 - cads_rescale_factor) * y
        else:
            print("Encountered NaN in cads rescaling. Skipping rescaling.")
    return y

# Tonemapping functions

def train_difference(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    diff_AB = a.float() - b.float()
    distance_A0 = torch.abs(b.float() - c.float())
    distance_A1 = torch.abs(b.float() - a.float())

    sum_distances = distance_A0 + distance_A1

    scale = torch.where(
        sum_distances != 0, distance_A1 / sum_distances, torch.tensor(0.0).float()
    )
    sign_scale = torch.sign(b.float() - c.float())
    scale = sign_scale * torch.abs(scale)
    new_diff = scale * torch.abs(diff_AB)
    return new_diff

def gated_thresholding(percentile: float, floor: float, t: Tensor) -> Tensor:
    """
    Args:
        percentile: float between 0.0 and 1.0. for example 0.995 would subject only the top 0.5%ile to clamping.
        t: [b, c, v] tensor in pixel or latent space (where v is the result of flattening w and h)
    """
    a = t.abs() # Magnitudes
    q = torch.quantile(a, percentile, dim=2) # Get clamp value via top % of magnitudes
    q.clamp_(min=floor)
    q = q.unsqueeze(2).expand(*t.shape)
    t = t.clamp(-q, q) # Clamp latent with magnitude value
    t = t / q
    return t

def dyn_thresh_gate(latent: Tensor, centered_magnitudes: Tensor, tonemap_percentile: float, floor: float, ceil: float):
    if centered_magnitudes.lt(torch.tensor(ceil, device=centered_magnitudes.device)).all().item(): # If the magnitudes are less than the ceiling
        return latent # Return the unmodified centered latent
    else:
        latent = gated_thresholding(tonemap_percentile, floor, latent) # If the magnitudes are higher than the ceiling
        return latent # Gated-dynamic thresholding by Birchlabs

def spatial_norm_thresholding(x0, value):
    # b c h w
    pow_x0 = torch.pow(torch.abs(x0), 2)
    s = pow_x0.mean(1, keepdim=True).sqrt().clamp(min=value)
    return x0 * (value / s)

def spatial_norm_chw_thresholding(x0, value):
    # b c h w
    pow_x0 = torch.pow(torch.abs(x0), 2)
    s = pow_x0.mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp(min=value)
    return x0 * (value / s)

# Contrast function

def contrast(x: Tensor):
    # Calculate the mean and standard deviation of the pixel values
    #mean = x.mean(dim=(1,2,3), keepdim=True)
    stddev = x.std(dim=(1,2,3), keepdim=True)
    # Scale the pixel values by the standard deviation
    scaled_pixels = (x) / stddev
    return scaled_pixels

def contrast_with_mean(x: Tensor):
    # Calculate the mean and standard deviation of the pixel values
    #mean = x.mean(dim=(2,3), keepdim=True)
    stddev = x.std(dim=(1,2,3), keepdim=True)
    diff_mean = ((x / stddev) - x).mean(dim=(1,2,3), keepdim=True)
    # Scale the pixel values by the standard deviation
    scaled_pixels = x / stddev
    return scaled_pixels - diff_mean

def center_latent(tensor): #https://birchlabs.co.uk/machine-learning#combating-mean-drift-in-cfg
    """Centers on 0 to combat CFG drift."""
    tensor = tensor - tensor.mean(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1).expand(tensor.shape)
    return tensor

def center_0channel(tensor): #https://birchlabs.co.uk/machine-learning#combating-mean-drift-in-cfg
    """Centers on 0 to combat CFG drift."""
    std_dev_0 = tensor[:, [0]].std()
    mean_0 = tensor[:, [0]].mean()
    mean_12 = tensor[:, [1,2]].mean()
    mean_3 = tensor[:, [3]].mean()

    #tensor[:, [0]] /= std_dev_0
    tensor[:, [0]] -= mean_0
    tensor[:, [0]] += torch.copysign(torch.pow(torch.abs(mean_0), 1.5), mean_0)
    #tensor[:, [1, 2]] -= tensor[:, [1, 2]].mean()
    tensor[:, [1, 2]] -= mean_12 * 0.5
    tensor[:, [3]] -= mean_3
    tensor[:, [3]] += torch.copysign(torch.pow(torch.abs(mean_3), 1.5), mean_3)
    return tensor# - tensor.mean(dim=(2,3), keepdim=True)

def channel_sharpen(tensor):
    """Centers on 0 to combat CFG drift."""
    flattened = tensor.flatten(2)
    flat_std = flattened.std(dim=(2)).unsqueeze(2).expand(flattened.shape)
    flattened *= flat_std
    flattened -= flattened.mean(dim=(2)).unsqueeze(2).expand(flattened.shape)
    flattened /= flat_std
    tensor = flattened.unflatten(2, tensor.shape[2:])
    return tensor


def center_012channel(tensor): #https://birchlabs.co.uk/machine-learning#combating-mean-drift-in-cfg
    """Centers on 0 to combat CFG drift."""
    curr_tens = tensor[:, [0,1,2]]
    tensor[:, [0,1,2]] -= curr_tens.mean()
    return tensor

def center_latent_perchannel(tensor): # Does nothing different than above
    """Centers on 0 to combat CFG drift."""
    flattened = tensor.flatten(2)
    flattened = flattened - flattened.mean(dim=(2)).unsqueeze(2).expand(flattened.shape)
    tensor = flattened.unflatten(2, tensor.shape[2:])
    return tensor

def center_latent_perchannel_with_magnitudes(tensor): # Does nothing different than above
    """Centers on 0 to combat CFG drift."""
    flattened = tensor.flatten(2)
    flattened_magnitude = (torch.linalg.vector_norm(flattened, dim=(2), keepdim=True) + 0.0000000001)
    flattened /= flattened_magnitude
    flattened = flattened - flattened.mean(dim=(2)).unsqueeze(2).expand(flattened.shape)
    flattened *= flattened_magnitude
    tensor = flattened.unflatten(2, tensor.shape[2:])
    return tensor

def center_latent_perchannel_with_decorrelate(tensor): # Decorrelates data, slight change, test and play with it.
    """Centers on 0 to combat CFG drift, preprocesses the latent with decorrelation"""
    tensor = decorrelate_data(tensor)
    flattened = tensor.flatten(2)
    flattened_magnitude = (torch.linalg.vector_norm(flattened, dim=(2), keepdim=True) + 0.0000000001)
    flattened /= flattened_magnitude
    flattened = flattened - flattened.mean(dim=(2)).unsqueeze(2).expand(flattened.shape)
    flattened *= flattened_magnitude
    tensor = flattened.unflatten(2, tensor.shape[2:])
    return tensor

def center_latent_median(tensor):
    flattened = tensor.flatten(2)
    median = flattened.median()
    scaled_data = (flattened - median)
    scaled_data = scaled_data.unflatten(2, tensor.shape[2:])
    return scaled_data

def divisive_normalization(image_tensor, neighborhood_size, threshold=1e-6):
    # Compute the local mean and local variance
    local_mean = F.avg_pool2d(image_tensor, neighborhood_size, stride=1, padding=neighborhood_size // 2, count_include_pad=False)
    local_mean_squared = local_mean**2
    
    local_variance = F.avg_pool2d(image_tensor**2, neighborhood_size, stride=1, padding=neighborhood_size // 2, count_include_pad=False) - local_mean_squared
    
    # Add a small value to prevent division by zero
    local_variance = local_variance + threshold
    
    # Apply divisive normalization
    normalized_tensor = image_tensor / torch.sqrt(local_variance)
    
    return normalized_tensor

def decorrelate_data(data):
    """flattened = tensor.flatten(2).squeeze(0) # this code aint shit, yo
    cov_matrix = torch.cov(flattened)
    sqrt_inv_cov_matrix = torch.linalg.inv(torch.sqrt(cov_matrix))
    decorrelated_tensor = torch.dot(flattened, sqrt_inv_cov_matrix.T)
    decorrelated_tensor = decorrelated_tensor.unflatten(2, tensor.shape[2:]).unsqueeze(0)"""

    # Reshape the 4D tensor to a 2D tensor for covariance calculation
    num_samples, num_channels, height, width = data.size()
    data_reshaped = data.view(num_samples, num_channels, -1)
    data_reshaped = data_reshaped - torch.mean(data_reshaped, dim=2, keepdim=True)

    # Compute covariance matrix
    cov_matrix = torch.matmul(data_reshaped, data_reshaped.transpose(1, 2)) / (height * width - 1)

    # Compute the inverse square root of the covariance matrix
    u, s, v = torch.svd(cov_matrix)
    sqrt_inv_cov_matrix = torch.matmul(u, torch.matmul(torch.diag_embed(1.0 / torch.sqrt(s)), v.transpose(1, 2)))

    # Reshape sqrt_inv_cov_matrix to match the dimensions of data_reshaped
    sqrt_inv_cov_matrix = sqrt_inv_cov_matrix.unsqueeze(0).expand(num_samples, -1, -1, -1)

    # Decorrelate the data
    decorrelated_data = torch.matmul(data_reshaped.transpose(1, 2), sqrt_inv_cov_matrix.transpose(2, 3))
    decorrelated_data = decorrelated_data.transpose(2, 3)
    
    # Reshape back to the original shape
    decorrelated_data = decorrelated_data.view(num_samples, num_channels, height, width)

    return decorrelated_data.to(data.device)

def get_low_frequency_noise(image: Tensor, threshold: float):
    # Convert image to Fourier domain
    fourier = torch.fft.fft2(image, dim=(-2, -1))  # Apply FFT along Height and Width dimensions
 
    # Compute the power spectrum
    power_spectrum = torch.abs(fourier) ** 2

    threshold = threshold ** 2

    # Drop low-frequency components
    mask = (power_spectrum < threshold).float()
    filtered_fourier = fourier * mask
    
    # Inverse transform back to spatial domain
    inverse_transformed = torch.fft.ifft2(filtered_fourier, dim=(-2, -1))  # Apply IFFT along Height and Width dimensions
    
    return inverse_transformed.real.to(image.device)

def spectral_modulation(image: Tensor, modulation_multiplier: float, spectral_mod_percentile: float): # Reference implementation by Clybius, 2023 :tm::c::r: (jk idc who uses it :3)
    # Convert image to Fourier domain
    fourier = torch.fft.fft2(image, dim=(-2, -1))  # Apply FFT along Height and Width dimensions
 
    log_amp = torch.log(torch.sqrt(fourier.real ** 2 + fourier.imag ** 2))

    quantile_low = torch.quantile(
        log_amp.abs().flatten(2),
        spectral_mod_percentile * 0.01,
        dim = 2
    ).unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)
    
    quantile_high = torch.quantile(
        log_amp.abs().flatten(2),
        1 - (spectral_mod_percentile * 0.01),
        dim = 2
    ).unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)

    # Increase low-frequency components
    mask_low = ((log_amp < quantile_low).float() + 1).clamp_(max=1.5) # If lower than low 5% quantile, set to 1.5, otherwise 1
    # Decrease high-frequency components
    mask_high = ((log_amp < quantile_high).float()).clamp_(min=0.5) # If lower than high 5% quantile, set to 1, otherwise 0.5
    filtered_fourier = fourier * ((mask_low * mask_high) ** modulation_multiplier) # Effectively
    
    # Inverse transform back to spatial domain
    inverse_transformed = torch.fft.ifft2(filtered_fourier, dim=(-2, -1))  # Apply IFFT along Height and Width dimensions
    
    return inverse_transformed.real.to(image.device)

def spectral_modulation_soft(image: Tensor, modulation_multiplier: float, spectral_mod_percentile: float): # Modified for soft quantile adjustment using a novel:tm::c::r: method titled linalg.
    # Convert image to Fourier domain
    fourier = torch.fft.fft2(image, dim=(-2, -1))  # Apply FFT along Height and Width dimensions
 
    log_amp = torch.log(torch.sqrt(fourier.real ** 2 + fourier.imag ** 2))

    quantile_low = torch.quantile(
        log_amp.abs().flatten(2),
        spectral_mod_percentile * 0.01,
        dim = 2
    ).unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)
    
    quantile_high = torch.quantile(
        log_amp.abs().flatten(2),
        1 - (spectral_mod_percentile * 0.01),
        dim = 2
    ).unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)

    quantile_max = torch.quantile(
        log_amp.abs().flatten(2),
        1,
        dim = 2
    ).unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)

    # Decrease high-frequency components
    mask_high = log_amp > quantile_high # If we're larger than 95th percentile

    additive_mult_high = torch.where(
        mask_high,
        1 - ((log_amp - quantile_high) / (quantile_max - quantile_high)).clamp_(max=0.5), # (1) - (0-1), where 0 is 95th %ile and 1 is 100%ile
        torch.tensor(1.0)
    )
    

    # Increase low-frequency components
    mask_low = log_amp < quantile_low
    additive_mult_low = torch.where(
        mask_low,
        1 + (1 - (log_amp / quantile_low)).clamp_(max=0.5), # (1) + (0-1), where 0 is 5th %ile and 1 is 0%ile
        torch.tensor(1.0)
    )
    
    mask_mult = ((additive_mult_low * additive_mult_high) ** modulation_multiplier).clamp_(min=0.05, max=20)
    #print(mask_mult)
    filtered_fourier = fourier * mask_mult
    
    # Inverse transform back to spatial domain
    inverse_transformed = torch.fft.ifft2(filtered_fourier, dim=(-2, -1))  # Apply IFFT along Height and Width dimensions
    
    return inverse_transformed.real.to(image.device)

def pyramid_noise_like(x, discount=0.9, generator=None, rand_source=random):
  b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
  u = torch.nn.Upsample(size=(w, h), mode='nearest-exact')
  noise = gen_like(torch.randn, x, generator=generator)
  for i in range(10):
    r = rand_source.random()*2+2 # Rather than always going 2x, 
    w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
    noise += u(torch.randn(b, c, w, h, generator=generator).to(x)) * discount**i
    if w==1 or h==1: break # Lowest resolution is 1x1
  return noise/noise.std() # Scaled back to roughly unit variance

import math
def dyn_cfg_modifier(conditioning, unconditioning, method, cond_scale, time_mult):
    match method:
        case "dyncfg-halfcosine":
            noise_pred = conditioning - unconditioning

            noise_pred_magnitude = (torch.linalg.vector_norm(noise_pred, dim=(1)) + 0.0000000001)[:,None]

            time = time_mult.item()
            time_factor = -(math.cos(0.5 * time * math.pi) / 2) + 1
            noise_pred_timescaled_magnitude = (torch.linalg.vector_norm(noise_pred * time_factor, dim=(1)) + 0.0000000001)[:,None]

            noise_pred /= noise_pred_magnitude
            noise_pred *= noise_pred_timescaled_magnitude
            return noise_pred
        case "dyncfg-halfcosine-mimic":
            noise_pred = conditioning - unconditioning

            noise_pred_magnitude = (torch.linalg.vector_norm(noise_pred, dim=(1)) + 0.0000000001)[:,None]

            time = time_mult.item()
            time_factor = -(math.cos(0.5 * time * math.pi) / 2) + 1

            latent = noise_pred

            mimic_latent = noise_pred * time_factor
            mimic_flattened = mimic_latent.flatten(2)
            mimic_means = mimic_flattened.mean(dim=2).unsqueeze(2)
            mimic_recentered = mimic_flattened - mimic_means
            mimic_abs = mimic_recentered.abs()
            mimic_max = mimic_abs.max(dim=2).values.unsqueeze(2)

            latent_flattened = latent.flatten(2)
            latent_means = latent_flattened.mean(dim=2).unsqueeze(2)
            latent_recentered = latent_flattened - latent_means
            latent_abs = latent_recentered.abs()
            latent_q = torch.quantile(latent_abs, 0.995, dim=2).unsqueeze(2)
            s = torch.maximum(latent_q, mimic_max)
            pred_clamped = noise_pred.flatten(2).clamp(-s, s)
            pred_normalized = pred_clamped / s
            pred_renorm = pred_normalized * mimic_max
            pred_uncentered = pred_renorm + latent_means
            noise_pred_degraded = pred_uncentered.unflatten(2, noise_pred.shape[2:])

            noise_pred /= noise_pred_magnitude

            noise_pred_timescaled_magnitude = (torch.linalg.vector_norm(noise_pred_degraded, dim=(1)) + 0.0000000001)[:,None]
            noise_pred *= noise_pred_timescaled_magnitude
            return noise_pred


class ModelSamplerLatentMegaModifier:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sharpness_multiplier": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                              "sharpness_method": (["anisotropic", "joint-anisotropic", "gaussian", "cas"], ),
                              "tonemap_multiplier": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                              "tonemap_method": (["reinhard", "reinhard_perchannel", "arctan", "quantile", "gated", "cfg-mimic", "spatial-norm"], ),
                              "tonemap_percentile": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.005}),
                              "contrast_multiplier": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                              "combat_method": (["subtract", "subtract_channels", "subtract_median", "sharpen"], ),
                              "combat_cfg_drift": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "rescale_cfg_phi": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "extra_noise_type": (["gaussian", "uniform", "perlin", "pink", "green", "pyramid"], ),
                              "extra_noise_method": (["add", "add_scaled", "speckle", "cads", "cads_rescaled", "cads_speckle", "cads_speckle_rescaled"], ),
                              "extra_noise_multiplier": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                              "extra_noise_lowpass": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 1}),
                              "divisive_norm_size": ("INT", {"default": 127, "min": 1, "max": 255, "step": 1}),
                              "divisive_norm_multiplier": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "spectral_mod_mode": (["hard_clamp", "soft_clamp"], ),
                              "spectral_mod_percentile": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.01}),
                              "spectral_mod_multiplier": ("FLOAT", {"default": 0.0, "min": -15.0, "max": 15.0, "step": 0.01}),
                              "affect_uncond": (["None", "Sharpness"], ),
                              "dyn_cfg_augmentation": (["None", "dyncfg-halfcosine", "dyncfg-halfcosine-mimic"], ),
                              },
                "optional": { "seed": ("INT", {"min": 0, "max": 0xffffffffffffffff})
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "mega_modify"

    CATEGORY = "clybNodes"

    def mega_modify(self, model, sharpness_multiplier, sharpness_method, tonemap_multiplier, tonemap_method, tonemap_percentile, contrast_multiplier, combat_method, combat_cfg_drift, rescale_cfg_phi, extra_noise_type, extra_noise_method, extra_noise_multiplier, extra_noise_lowpass, divisive_norm_size, divisive_norm_multiplier, spectral_mod_mode, spectral_mod_percentile, spectral_mod_multiplier, affect_uncond, dyn_cfg_augmentation, seed=None):
        gen = None
        rand = random
        if seed is not None:
            gen = torch.Generator(device='cpu')
            rand = random.Random()
            gen.manual_seed(seed)
            rand.seed(seed)

        def modify_latent(args):
            x_input = args["input"]
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            timestep = model.model.model_sampling.timestep(args["timestep"])
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            #print(model.model.model_sampling.timestep(timestep))

            x = x_input / (sigma * sigma + 1.0)
            cond = ((x - (x_input - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
            uncond = ((x - (x_input - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)

            noise_pred = (cond - uncond)

            # Extra noise
            if extra_noise_multiplier > 0:
                match extra_noise_type:
                    case "gaussian":
                        extra_noise = gen_like(torch.randn, cond, generator=gen)
                    case "uniform":
                        extra_noise = (gen_like(torch.rand, cond, generator=gen) - 0.5) * 2 * 1.73
                    case "perlin":
                        cond_size_0 = cond.size(dim=2)
                        cond_size_1 = cond.size(dim=3)
                        extra_noise = perlin_noise(grid_shape=(cond_size_0, cond_size_1), out_shape=(cond_size_0, cond_size_1), batch_size=4, generator=gen).to(cond.device).unsqueeze(0)
                        mean = torch.mean(extra_noise)
                        std = torch.std(extra_noise)

                        extra_noise.sub_(mean).div_(std)
                    case "pink":
                        extra_noise = generate_1f_noise(cond, 2, extra_noise_multiplier, generator=gen).to(cond.device)
                        mean = torch.mean(extra_noise)
                        std = torch.std(extra_noise)

                        extra_noise.sub_(mean).div_(std)
                    case "green":
                        cond_size_0 = cond.size(dim=2)
                        cond_size_1 = cond.size(dim=3)
                        extra_noise = green_noise(cond_size_0, cond_size_1, generator=gen).to(cond.device)
                        mean = torch.mean(extra_noise)
                        std = torch.std(extra_noise)

                        extra_noise.sub_(mean).div_(std)
                    case "pyramid":
                        extra_noise = pyramid_noise_like(cond)
                
                if extra_noise_lowpass > 0:
                    extra_noise = get_low_frequency_noise(extra_noise, extra_noise_lowpass)

                alpha_noise = 1.0 - (timestep / 999.0)[:, None, None, None].clone() # Get alpha multiplier, lower alpha at high sigmas/high noise
                alpha_noise *= 0.001 * extra_noise_multiplier # User-input and weaken the strength so we don't annihilate the latent.
                match extra_noise_method:
                    case "add":
                        cond = cond + extra_noise * alpha_noise
                        uncond = uncond - extra_noise * alpha_noise
                    case "add_scaled":
                        cond = cond + train_difference(cond, extra_noise, cond) * alpha_noise
                        uncond = uncond - train_difference(uncond, extra_noise, uncond) * alpha_noise
                    case "speckle":
                        cond = cond + cond * extra_noise * alpha_noise
                        uncond = uncond - uncond * extra_noise * alpha_noise
                    case "cads":
                        cond = add_cads_custom_noise(cond, extra_noise, timestep, 0.6, 0.9, extra_noise_multiplier / 100., 1, False)
                        uncond = add_cads_custom_noise(uncond, extra_noise, timestep, 0.6, 0.9, extra_noise_multiplier / 100., 1, False)
                    case "cads_rescaled":
                        cond = add_cads_custom_noise(cond, extra_noise, timestep, 0.6, 0.9, extra_noise_multiplier / 100., 1, True)
                        uncond = add_cads_custom_noise(uncond, extra_noise, timestep, 0.6, 0.9, extra_noise_multiplier / 100., 1, True)
                    case "cads_speckle":
                        cond = add_cads_custom_noise(cond, extra_noise * cond, timestep, 0.6, 0.9, extra_noise_multiplier / 100., 1, False)
                        uncond = add_cads_custom_noise(uncond, extra_noise * uncond, timestep, 0.6, 0.9, extra_noise_multiplier / 100., 1, False)
                    case "cads_speckle_rescaled":
                        cond = add_cads_custom_noise(cond, extra_noise * cond, timestep, 0.6, 0.9, extra_noise_multiplier / 100., 1, True)
                        uncond = add_cads_custom_noise(uncond, extra_noise * uncond, timestep, 0.6, 0.9, extra_noise_multiplier / 100., 1, True)
                    case _:
                        print("Haven't heard of a noise method named like that before... (Couldn't find method)")

            if sharpness_multiplier > 0.0 or sharpness_multiplier < 0.0:
                match sharpness_method:
                    case "anisotropic":
                        degrade_func = bilateral_blur
                    case "joint-anisotropic":
                        degrade_func = lambda img: joint_bilateral_blur(img, (img - torch.mean(img, dim=(1, 2, 3), keepdim=True)) / torch.std(img, dim=(1, 2, 3), keepdim=True), 13, 3.0, 3.0, "reflect", "l1")
                    case "gaussian":
                        degrade_func = gaussian_filter_2d
                    case "cas":
                        degrade_func = lambda image: contrast_adaptive_sharpening(image, amount=sigma.clamp(max=1.00).item())
                    case _:
                        print("For some reason, the sharpness filter could not be found.")
                # Sharpness
                alpha = 1.0 - (timestep / 999.0)[:, None, None, None].clone() # Get alpha multiplier, lower alpha at high sigmas/high noise
                alpha *= 0.001 * sharpness_multiplier # User-input and weaken the strength so we don't annihilate the latent.
                cond = degrade_func(cond) * alpha + cond * (1.0 - alpha) # Mix the modified latent with the existing latent by the alpha
                if affect_uncond == "Sharpness":
                    uncond = degrade_func(uncond) * alpha + uncond * (1.0 - alpha)

            time_mult = 1.0 - (timestep / 999.0)[:, None, None, None].clone()
            noise_pred_degraded = (cond - uncond) if dyn_cfg_augmentation == "None" else dyn_cfg_modifier(cond, uncond, dyn_cfg_augmentation, cond_scale, time_mult) # New noise pred

            # After this point, we use `noise_pred_degraded` instead of just `cond` for the final set of calculations

            # Tonemap noise
            if tonemap_multiplier == 0:
                new_magnitude = 1.0
            else:
                match tonemap_method:
                    case "reinhard":
                        noise_pred_vector_magnitude = (torch.linalg.vector_norm(noise_pred_degraded, dim=(1)) + 0.0000000001)[:,None]
                        noise_pred_degraded /= noise_pred_vector_magnitude

                        mean = torch.mean(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
                        std = torch.std(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)

                        top = (std * 3 * (100 / tonemap_percentile) + mean) * tonemap_multiplier

                        noise_pred_vector_magnitude *= (1.0 / top)
                        new_magnitude = noise_pred_vector_magnitude / (noise_pred_vector_magnitude + 1.0)
                        new_magnitude *= top

                        noise_pred_degraded *= new_magnitude
                    case "reinhard_perchannel": # Testing the flatten strategy
                        flattened = noise_pred_degraded.flatten(2)
                        noise_pred_vector_magnitude = (torch.linalg.vector_norm(flattened, dim=(2), keepdim=True) + 0.0000000001)
                        flattened /= noise_pred_vector_magnitude

                        mean = torch.mean(noise_pred_vector_magnitude, dim=(2), keepdim=True)

                        top = (3 * (100 / tonemap_percentile) + mean) * tonemap_multiplier

                        noise_pred_vector_magnitude *= (1.0 / top)

                        new_magnitude = noise_pred_vector_magnitude / (noise_pred_vector_magnitude + 1.0)
                        new_magnitude *= top

                        flattened *= new_magnitude
                        noise_pred_degraded = flattened.unflatten(2, noise_pred_degraded.shape[2:])
                    case "arctan":
                        noise_pred_vector_magnitude = (torch.linalg.vector_norm(noise_pred_degraded, dim=(1)) + 0.0000000001)[:,None]
                        noise_pred_degraded /= noise_pred_vector_magnitude

                        noise_pred_degraded = (torch.arctan(noise_pred_degraded * tonemap_multiplier) * (1 / tonemap_multiplier)) + (noise_pred_degraded * (100 - tonemap_percentile) / 100)

                        noise_pred_degraded *= noise_pred_vector_magnitude
                    case "quantile":
                        s: FloatTensor = torch.quantile(
                            (uncond + noise_pred_degraded * cond_scale).flatten(start_dim=1).abs(),
                            tonemap_percentile / 100,
                            dim = -1
                        ) * tonemap_multiplier
                        s.clamp_(min = 1.)
                        s = s.reshape(*s.shape, 1, 1, 1)
                        noise_pred_degraded = noise_pred_degraded.clamp(-s, s) / s
                    case "gated": # https://birchlabs.co.uk/machine-learning#dynamic-thresholding-latents so based,.,.,....,
                        latent_scale = model.model.latent_format.scale_factor

                        latent = uncond + noise_pred_degraded * cond_scale # Get full latent from CFG formula
                        latent /= latent_scale # Divide full CFG by latent scale (~0.13 for sdxl)
                        flattened = latent.flatten(2)
                        means = flattened.mean(dim=2).unsqueeze(2)
                        centered_magnitudes = (flattened - means).abs().max() # Get highest magnitude of full CFG
                        
                        flattened_pred = (noise_pred_degraded / latent_scale).flatten(2)

                        floor = 3.0560
                        ceil = 42. * tonemap_multiplier # as is the answer to life, unless you modify the multiplier cuz u aint a believer in life


                        thresholded_latent = dyn_thresh_gate(flattened_pred, centered_magnitudes, tonemap_percentile / 100., floor, ceil) # Threshold if passes ceil
                        thresholded_latent = thresholded_latent.unflatten(2, noise_pred_degraded.shape[2:])
                        noise_pred_degraded = thresholded_latent * latent_scale # Rescale by latent
                    case "cfg-mimic":
                        latent = noise_pred_degraded

                        mimic_latent = noise_pred_degraded * tonemap_multiplier
                        mimic_flattened = mimic_latent.flatten(2)
                        mimic_means = mimic_flattened.mean(dim=2).unsqueeze(2)
                        mimic_recentered = mimic_flattened - mimic_means
                        mimic_abs = mimic_recentered.abs()
                        mimic_max = mimic_abs.max(dim=2).values.unsqueeze(2)

                        latent_flattened = latent.flatten(2)
                        latent_means = latent_flattened.mean(dim=2).unsqueeze(2)
                        latent_recentered = latent_flattened - latent_means
                        latent_abs = latent_recentered.abs()
                        latent_q = torch.quantile(latent_abs, tonemap_percentile / 100., dim=2).unsqueeze(2)
                        s = torch.maximum(latent_q, mimic_max)
                        pred_clamped = noise_pred_degraded.flatten(2).clamp(-s, s)
                        pred_normalized = pred_clamped / s
                        pred_renorm = pred_normalized * mimic_max
                        pred_uncentered = pred_renorm + mimic_means # Personal choice to re-mean from the mimic here... should be latent_means.
                        noise_pred_degraded = pred_uncentered.unflatten(2, noise_pred_degraded.shape[2:])
                    case "spatial-norm":
                        #time = (1.0 - (timestep / 999.0)[:, None, None, None].clone().item())
                        #time = -(math.cos(time * math.pi) / (3)) + (2/3) # 0.33333 to 1.0, half cosine
                        noise_pred_degraded = spatial_norm_chw_thresholding(noise_pred_degraded, tonemap_multiplier / 2 / cond_scale)
                    case _:
                        print("Could not tonemap, for the method was not found.")

            # Spectral Modification
            if spectral_mod_multiplier > 0 or spectral_mod_multiplier < 0:
                #alpha = 1. - (timestep / 999.0)[:, None, None, None].clone() # Get alpha multiplier, lower alpha at high sigmas/high noise
                #alpha = spectral_mod_multiplier# User-input and weaken the strength so we don't annihilate the latent.
                match spectral_mod_mode:
                    case "hard_clamp":
                        modulation_func = spectral_modulation
                    case "soft_clamp":
                        modulation_func = spectral_modulation_soft
                modulation_diff = modulation_func(noise_pred_degraded, spectral_mod_multiplier, spectral_mod_percentile) - noise_pred_degraded
                noise_pred_degraded += modulation_diff

            if contrast_multiplier > 0 or contrast_multiplier < 0:
                contrast_func = contrast
                # Contrast, after tonemapping, to ensure user-set contrast is expected to behave similarly across tonemapping settings
                alpha = 1.0 - (timestep / 999.0)[:, None, None, None].clone()
                alpha *= 0.001 * contrast_multiplier
                noise_pred_degraded = contrast_func(noise_pred_degraded) * alpha + (noise_pred_degraded) * (1.0 - alpha) # Temporary fix for contrast is to add the input? Maybe? It just doesn't work like before...

            # Rescale CFG
            if rescale_cfg_phi == 0:
                x_final = uncond + noise_pred_degraded * cond_scale
            else:
                x_cfg = uncond + noise_pred_degraded * cond_scale
                ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
                ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)

                x_rescaled = x_cfg * (ro_pos / ro_cfg)
                x_final = rescale_cfg_phi * x_rescaled + (1.0 - rescale_cfg_phi) * x_cfg

            if combat_cfg_drift > 0 or combat_cfg_drift < 0:
                alpha = (1. - (timestep / 999.0)[:, None, None, None].clone())
                alpha ** 0.025 # Alpha might as well be 1, but we want to protect the first steps (?).
                alpha = alpha.clamp_(max=1)
                match combat_method:
                    case "subtract":
                        combat_drift_func = center_latent_perchannel
                        alpha *= combat_cfg_drift 
                    case "subtract_channels":
                        combat_drift_func = center_0channel
                        alpha *= combat_cfg_drift
                    case "subtract_median":
                        combat_drift_func = center_latent_median
                        alpha *= combat_cfg_drift
                    case "sharpen":
                        combat_drift_func = channel_sharpen
                        alpha *= combat_cfg_drift
                x_final = combat_drift_func(x_final) * alpha + x_final * (1.0 - alpha) # Mix the modified latent with the existing latent by the alpha
            
            if divisive_norm_multiplier > 0:
                alpha = 1. - (timestep / 999.0)[:, None, None, None].clone()
                alpha ** 0.025 # Alpha might as well be 1, but we want to protect the beginning steps (?).
                alpha *= divisive_norm_multiplier
                high_noise = divisive_normalization(x_final, (divisive_norm_size * 2) + 1)
                x_final = high_noise * alpha + x_final * (1.0 - alpha)


            return x_input - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5) # General formula for CFG. uncond + (cond - uncond) * cond_scale

        m = model.clone()
        m.set_model_sampler_cfg_function(modify_latent)
        return (m, )