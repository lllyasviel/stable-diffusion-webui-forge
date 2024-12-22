# Taken from https://github.com/comfyanonymous/ComfyUI
# This file is only for reference, and not used in the backend or runtime.


import math

from scipy import integrate
import torch
from torch import nn
import torchsde
from tqdm.auto import trange, tqdm

from ldm_patched.modules import utils
from ldm_patched.k_diffusion import deis
import ldm_patched.modules.model_patcher
import ldm_patched.modules.model_sampling
import torchdiffeq
import modules.shared
from torch import no_grad, FloatTensor
from typing import Protocol, Optional, Dict, Any, TypedDict, NamedTuple, List
from itertools import pairwise
from ldm_patched.modules.model_sampling import CONST
from modules.shared import opts
import numpy as np

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1., device='cpu'):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)

# align your steps
def get_sigmas_ays(n, sigma_min, sigma_max, is_sdxl=False, device='cpu'):
    # https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
    def loglinear_interp(t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        xs = torch.linspace(0, 1, len(t_steps))
        ys = torch.log(torch.tensor(t_steps[::-1]))

        new_xs = torch.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)

        interped_ys = torch.exp(torch.tensor(new_ys)).numpy()[::-1].copy()
        return interped_ys

    if is_sdxl:
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
    else:
        # Default to SD 1.5 sigmas.
        sigmas = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)

def get_sigmas_ays_gits(n, sigma_min, sigma_max, is_sdxl=False, device='cpu'):
    def loglinear_interp(t_steps, num_steps):
        xs = torch.linspace(0, 1, len(t_steps))
        ys = torch.log(torch.tensor(t_steps[::-1]))
        new_xs = torch.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)
        interped_ys = torch.exp(torch.tensor(new_ys)).numpy()[::-1].copy()
        return interped_ys

    if is_sdxl:
        sigmas = [14.615, 4.734, 2.567, 1.529, 0.987, 0.652, 0.418, 0.268, 0.179, 0.127, 0.029]
    else:
        sigmas = [14.615, 4.617, 2.507, 1.236, 0.702, 0.402, 0.240, 0.156, 0.104, 0.094, 0.029]

    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)

def get_sigmas_ays_11steps(n, sigma_min, sigma_max, is_sdxl=False, device='cpu'):
    # This is the same as the original AYS
    return get_sigmas_ays(n, sigma_min, sigma_max, is_sdxl, device)

def get_sigmas_ays_32steps(n, sigma_min, sigma_max, is_sdxl=False, device='cpu'):
    def loglinear_interp(t_steps, num_steps):
        xs = torch.linspace(0, 1, len(t_steps))
        ys = torch.log(torch.tensor(t_steps[::-1]))
        new_xs = torch.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)
        interped_ys = torch.exp(torch.tensor(new_ys)).numpy()[::-1].copy()
        return interped_ys
    
    if is_sdxl:
        sigmas = [14.61500000000000000, 11.14916180000000000, 8.505221270000000000, 6.488271510000000000, 5.437074020000000000, 4.603986190000000000, 3.898547040000000000, 3.274074570000000000, 2.743965270000000000, 2.299686590000000000, 1.954485140000000000, 1.671087150000000000, 1.428781520000000000, 1.231810090000000000, 1.067896490000000000, 0.925794430000000000, 0.802908860000000000, 0.696601210000000000, 0.604369030000000000, 0.528525520000000000, 0.467733440000000000, 0.413933790000000000, 0.362581860000000000, 0.310085170000000000, 0.265189250000000000, 0.223264610000000000, 0.176538770000000000, 0.139591920000000000, 0.105873810000000000, 0.055193690000000000, 0.028773340000000000, 0.015000000000000000]
    else:
        sigmas = [14.61500000000000000, 11.23951352000000000, 8.643630810000000000, 6.647294240000000000, 5.572508620000000000, 4.716485460000000000, 3.991960650000000000, 3.519560900000000000, 3.134904660000000000, 2.792287880000000000, 2.487736280000000000, 2.216638650000000000, 1.975083510000000000, 1.779317200000000000, 1.614753350000000000, 1.465409530000000000, 1.314849000000000000, 1.166424970000000000, 1.034755470000000000, 0.915737440000000000, 0.807481690000000000, 0.712023610000000000, 0.621739000000000000, 0.530652020000000000, 0.452909600000000000, 0.374914550000000000, 0.274618190000000000, 0.201152900000000000, 0.141058730000000000, 0.066828810000000000, 0.031661210000000000, 0.015000000000000000]
    
    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)
    
    return torch.FloatTensor(sigmas).to(device)

def cosine_scheduler(n, sigma_min, sigma_max, device='cpu'):
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        for x in range(n):
            p = x / (n-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas[x] = C
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def cosexpblend_scheduler(n, sigma_min, sigma_max, device='cpu'):
    sigmas = []
    if n == 1:
        sigmas.append(sigma_max ** 0.5)
    else:
        K = (sigma_min / sigma_max)**(1/(n-1))
        E = sigma_max
        for x in range(n):
            p = x / (n-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas.append(C + p * (E - C))
            E *= K
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)

def phi_scheduler(n, sigma_min, sigma_max, device='cpu'):
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        phi = (1 + 5**0.5) / 2
        for x in range(n):
            sigmas[x] = sigma_min + (sigma_max-sigma_min)*((1-x/(n-1))**(phi*phi))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_laplace(n, sigma_min, sigma_max, device='cpu'):
    mu = 0.
    beta = 0.5
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_karras_dynamic(n, sigma_min, sigma_max, device='cpu'):
    rho = 7.
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.zeros_like(ramp)
    for i in range(n):
        sigmas[i] = (max_inv_rho + ramp[i] * (min_inv_rho - max_inv_rho)) ** (math.cos(i*math.tau/n)*2+rho) 
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_sinusoidal_sf(n, sigma_min, sigma_max, sf=3.5, device='cpu'):
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (1 - torch.sin(torch.pi / 2 * x)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_invcosinusoidal_sf(n, sigma_min, sigma_max, sf=3.5, device='cpu'):
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (0.5*(torch.cos(x * math.pi) + 1)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_react_cosinusoidal_dynsf(n, sigma_min, sigma_max, sf=2.15, device='cpu'):
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min+(sigma_max-sigma_min)*(torch.cos(x*(torch.pi/2))))/sigma_max
    sigmas = sigmas**(sf*(n*x/n))
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)

def get_sigmas_laplace(n, sigma_min, sigma_max, mu=0., beta=0.5, device='cpu'):
    """Constructs the noise schedule proposed by Tiankai et al. (2024). """
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return sigmas


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=None):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    eta = eta if eta is not None else opts.ancestral_eta
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)

@torch.no_grad()
def sample_dpmpp_2m_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    t_fn = lambda sigma: sigma.log().neg()
    old_uncond_denoised = None
    uncond_denoised = None
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]
    
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            denoised_mix = -torch.exp(-h) * uncond_denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_uncond_denoised)
        x = denoised + denoised_mix + torch.exp(-h) * x
        old_uncond_denoised = uncond_denoised
    return x

ADAPTIVE_SOLVERS = {"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"}
FIXED_SOLVERS = {"euler", "midpoint", "rk4", "heun3", "explicit_adams", "implicit_adams"}
ALL_SOLVERS = list(ADAPTIVE_SOLVERS | FIXED_SOLVERS)
ALL_SOLVERS.sort()
class ODEFunction:
    def __init__(self, model, t_min, t_max, n_steps, is_adaptive, extra_args=None, callback=None):
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.callback = callback
        self.t_min = t_min.item()
        self.t_max = t_max.item()
        self.n_steps = n_steps
        self.is_adaptive = is_adaptive
        self.step = 0

        if is_adaptive:
            self.pbar = tqdm(
                total=100,
                desc="solve",
                unit="%",
                leave=False,
                position=1
            )
        else:
            self.pbar = tqdm(
                total=n_steps,
                desc="solve",
                leave=False,
                position=1
            )

    def __call__(self, t, y):
        if t <= 1e-5:
            return torch.zeros_like(y)

        denoised = self.model(y.unsqueeze(0), t.unsqueeze(0), **self.extra_args)
        return (y - denoised.squeeze(0)) / t

    def _callback(self, t0, y0, step):
        if self.callback is not None:
            y0 = y0.unsqueeze(0)

            self.callback({
                "x": y0,
                "i": step,
                "sigma": t0,
                "sigma_hat": t0,
                "denoised": y0, # for a bad latent preview
            })

    def callback_step(self, t0, y0, dt):
        if self.is_adaptive:
            return

        self._callback(t0, y0, self.step)

        self.pbar.update(1)
        self.step += 1

    def callback_accept_step(self, t0, y0, dt):
        if not self.is_adaptive:
            return

        progress = (self.t_max - t0.item()) / (self.t_max - self.t_min)

        self._callback(t0, y0, round((self.n_steps - 1) * progress))

        new_step = round(100 * progress)
        self.pbar.update(new_step - self.step)
        self.step = new_step

    def reset(self):
        self.step = 0
        self.pbar.reset()
class Rescaler:
    def __init__(self, model, x, mode, **extra_args):
        self.model = model
        self.x = x
        self.mode = mode
        self.extra_args = extra_args

        self.latent_image, self.noise = model.latent_image, model.noise
        self.denoise_mask = self.extra_args.get("denoise_mask", None)

    def __enter__(self):
        if self.latent_image is not None:
            self.model.latent_image = torch.nn.functional.interpolate(input=self.latent_image, size=self.x.shape[2:4], mode=self.mode)
        if self.noise is not None:
            self.model.noise = torch.nn.functional.interpolate(input=self.latent_image, size=self.x.shape[2:4], mode=self.mode)
        if self.denoise_mask is not None:
            self.extra_args["denoise_mask"] = torch.nn.functional.interpolate(input=self.denoise_mask, size=self.x.shape[2:4], mode=self.mode)

        return self

    def __exit__(self, type, value, traceback):
        del self.model.latent_image, self.model.noise
        self.model.latent_image, self.model.noise = self.latent_image, self.noise
class ODESampler:
    def __init__(self, solver, rtol, atol, max_steps):
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    @torch.no_grad()
    def __call__(self, model, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None):
        t_max = sigmas.max()
        t_min = sigmas.min()
        n_steps = len(sigmas)

        if self.solver in FIXED_SOLVERS:
            t = sigmas
            is_adaptive = False
        else:
            t = torch.stack([t_max, t_min])
            is_adaptive = True

        ode = ODEFunction(model, t_min, t_max, n_steps, is_adaptive=is_adaptive, callback=callback, extra_args=extra_args)

        samples = torch.empty_like(x)
        for i in trange(x.shape[0], desc=self.solver, disable=disable):
            ode.reset()

            samples[i] = torchdiffeq.odeint(
                ode,
                x[i],
                t,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver,
                options={
                    "min_step": 1e-5,
                    "max_num_steps": self.max_steps,
                    "dtype": torch.float32 if torch.backends.mps.is_available() else torch.float64
                }
            )[-1]

        if callback is not None:
            callback({
                "x": samples,
                "i": n_steps - 1,
                "sigma": t_min,
                "sigma_hat": t_min,
                "denoised": samples, # only accurate if t_min = 0, for now
            })

        return samples


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs) for s in seed]
        else:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if self.cpu_tree:
            w = torch.stack([tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device) for tree in self.trees]) * (self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    s_churn = modules.shared.opts.euler_og_s_churn
    s_tmin = modules.shared.opts.euler_og_s_tmin
    s_noise = modules.shared.opts.euler_og_s_noise
    s_tmax = float('inf')

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    if hasattr(model, 'model_sampling') and isinstance(model.model_sampling, CONST):
        return sample_euler_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)
    """Ancestral sampling with Euler method steps."""
    eta = modules.shared.opts.euler_ancestral_og_eta
    s_noise = modules.shared.opts.euler_ancestral_og_s_noise

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            x = denoised
        else:
            d = to_d(x, sigmas[i], denoised)
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_euler_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        # sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1]**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2)**0.5
            # Euler method
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            dt = sigma_down - sigmas[i]
            x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            # r = torch.sinh(1 + (2 - eta) * (t_next - t) / (t - t_fn(sigma_up))) works only on non-cfgpp, weird
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_churn = modules.shared.opts.heun_og_s_churn
    s_tmin = modules.shared.opts.heun_og_s_tmin
    s_noise = modules.shared.opts.heun_og_s_noise
    s_tmax = float('inf')

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_dpm_2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_dpm_2_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        cur_order = min(i + 1, order)
        coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
        x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x


class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control."""
    def __init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927."""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_fast(self, x, t_start, t_end, nfe, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if not t_end > t_start and eta:
            raise ValueError('eta must be 0 for reverse sampling')

        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else:
                t_next_, su = t_next, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
            denoised = x - self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': i, 't': ts[i], 't_up': t, 'denoised': denoised})

            if orders[i] == 1:
                x, eps_cache = self.dpm_solver_1_step(x, t, t_next_, eps_cache=eps_cache)
            elif orders[i] == 2:
                x, eps_cache = self.dpm_solver_2_step(x, t, t_next_, eps_cache=eps_cache)
            else:
                x, eps_cache = self.dpm_solver_3_step(x, t, t_next_, eps_cache=eps_cache)

            x = x + su * s_noise * noise_sampler(self.sigma(t), self.sigma(t_next))

        return x

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if order not in {2, 3}:
            raise ValueError('order should be 2 or 3')
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError('eta must be 0 for reverse sampling')
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else:
                t_, su = t, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
            denoised = x - self.sigma(s) * eps

            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(x, s, t_, eps_cache=eps_cache)
            else:
                x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info['n_accept'] += 1
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

        return x, info


@torch.no_grad()
def sample_dpm_fast(model, x, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None, eta=0., s_noise=1., noise_sampler=None):
    """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(total=n, disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        return dpm_solver.dpm_solver_fast(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), n, eta, s_noise, noise_sampler)


@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    if return_info:
        return x, info
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    eta = modules.shared.opts.dpm_2s_ancestral_og_eta
    s_noise = modules.shared.opts.dpm_2s_ancestral_og_s_noise
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """DPM-Solver++ (stochastic)."""
    eta = modules.shared.opts.dpmpp_sde_og_eta
    s_noise = modules.shared.opts.dpmpp_sde_og_s_noise
    r = modules.shared.opts.dpmpp_sde_og_r
    
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)
            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """DPM-Solver++(2M) SDE."""
    eta = modules.shared.opts.dpmpp_2m_sde_og_eta
    s_noise = modules.shared.opts.dpmpp_2m_sde_og_s_noise
    solver_type = modules.shared.opts.dpmpp_2m_sde_og_solver_type
    
    if len(sigmas) <= 1:
        return x

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h
            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised
            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)
            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise
        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """DPM-Solver++(3M) SDE."""
    eta = modules.shared.opts.dpmpp_3m_sde_og_eta
    s_noise = modules.shared.opts.dpmpp_3m_sde_og_s_noise
    
    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)
            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised
            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d
            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise
        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_3m_sde(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler)

@torch.no_grad()
def sample_dpmpp_2m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_2m_sde(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, solver_type=solver_type)

@torch.no_grad()
def sample_dpmpp_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_sde(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, r=r)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu

def generic_step_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0), sigmas[i], sigmas[i + 1], (x - denoised) / sigmas[i], noise_sampler)
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


@torch.no_grad()
def sample_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step)

@torch.no_grad()
def sample_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x = model.inner_model.inner_model.model_sampling.noise_scaling(sigmas[i + 1], noise_sampler(sigmas[i], sigmas[i + 1]), x)
    return x



@torch.no_grad()
def sample_heunpp2(model, x, sigmas, extra_args=None, callback=None, disable=None):
    s_churn = modules.shared.opts.heunpp2_s_churn
    s_tmin = modules.shared.opts.heunpp2_s_tmin
    s_noise = modules.shared.opts.heunpp2_s_noise
    s_tmax = float('inf')
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    s_end = sigmas[-1]
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == s_end:
            # Euler method
            x = x + d * dt
        elif sigmas[i + 2] == s_end:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            w = 2 * sigmas[0]
            w2 = sigmas[i+1]/w
            w1 = 1 - w2
            d_prime = d * w1 + d_2 * w2
            x = x + d_prime * dt
        else:
            # Heun++
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            dt_2 = sigmas[i + 2] - sigmas[i + 1]
            x_3 = x_2 + d_2 * dt_2
            denoised_3 = model(x_3, sigmas[i + 2] * s_in, **extra_args)
            d_3 = to_d(x_3, sigmas[i + 2], denoised_3)
            w = 3 * sigmas[0]
            w2 = sigmas[i + 1] / w
            w3 = sigmas[i + 2] / w
            w1 = 1 - w2 - w3
            d_prime = w1 * d + w2 * d_2 + w3 * d_3
            x = x + d_prime * dt
    return x

#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
def sample_ipndm(model, x, sigmas, extra_args=None, callback=None, disable=None):
    max_order = modules.shared.opts.ipndm_max_order
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    x_next = x
    buffer_model = []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        x_cur = x_next
        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i+1)
        if order == 1:      # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:    # Use two history points.
            x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:    # Use three history points.
            x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur
        else:
            buffer_model.append(d_cur)
    return x_next

#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
def sample_ipndm_v(model, x, sigmas, extra_args=None, callback=None, disable=None):
    max_order = modules.shared.opts.ipndm_v_max_order
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    x_next = x
    t_steps = sigmas
    buffer_model = []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        x_cur = x_next
        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i+1)
        if order == 1:      # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            coeff1 = (2 + (h_n / h_n_1)) / 2
            coeff2 = -(h_n / h_n_1) / 2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1])
        elif order == 3:    # Use two history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp
            coeff3 = temp * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2])
        elif order == 4:    # Use three history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            h_n_3 = (t_steps[i-2] - t_steps[i-3])
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                   * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2] + coeff4 * buffer_model[-3])
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
    return x_next

#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
@torch.no_grad()
def sample_deis(model, x, sigmas, extra_args=None, callback=None, disable=None):
    max_order = modules.shared.opts.deis_max_order
    deis_mode = modules.shared.opts.deis_mode
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    x_next = x
    t_steps = sigmas
    coeff_list = deis.get_deis_coeff_list(t_steps, max_order, deis_mode=deis_mode)
    buffer_model = []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        x_cur = x_next
        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i+1)
        if t_next <= 0:
            order = 1
        if order == 1:          # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:        # Use one history point.
            coeff_cur, coeff_prev1 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1]
        elif order == 3:        # Use two history points.
            coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
        elif order == 4:        # Use three history points.
            coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
    return x_next

@torch.no_grad()
def sample_euler_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, temp[0])
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        # Euler method
        x = denoised + d * sigmas[i + 1]
    return x

@torch.no_grad()
def sample_euler_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    eta = modules.shared.opts.euler_ancestral_cfg_pp_eta
    s_noise = modules.shared.opts.euler_ancestral_cfg_pp_s_noise
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], temp[0])
        # Euler method
        x = denoised + d * sigma_down
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps and CFG++."""
    eta = modules.shared.opts.dpmpp_2s_ancestral_cfg_pp_eta
    s_noise = modules.shared.opts.dpmpp_2s_ancestral_cfg_pp_s_noise
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            d = to_d(x, sigmas[i], temp[0])
            x = denoised_2 + d * sigma_down
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp_dyn(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=None, s_noise=None, noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    eta = modules.shared.opts.dpmpp_2s_ancestral_dyn_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_2s_ancestral_dyn_s_noise if s_noise is None else s_noise
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            dt = sigma_down - sigmas[i]
            x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = torch.sinh(1 + (2 - eta) * (t_next - t) / (t - t_fn(sigma_up)))
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda lbda: (lbda.exp() + 1) ** -1
    lambda_fn = lambda sigma: ((1-sigma)/sigma).log()

    # logged_x = x.unsqueeze(0)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        downstep_ratio = 1 + (sigmas[i+1]/sigmas[i] - 1) * eta
        sigma_down = sigmas[i+1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i+1]
        alpha_down = 1 - sigma_down
        renoise_coeff = (sigmas[i+1]**2 - sigma_down**2*alpha_ip1**2/alpha_down**2)**0.5
        # sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            if sigmas[i] == 1.0:
                sigma_s = 0.9999
            else:
                t_i, t_down = lambda_fn(sigmas[i]), lambda_fn(sigma_down)
                r = 1 / 2
                h = t_down - t_i
                s = t_i + r * h
                sigma_s = sigma_fn(s)
            # sigma_s = sigmas[i+1]
            sigma_s_i_ratio = sigma_s / sigmas[i]
            u = sigma_s_i_ratio * x + (1 - sigma_s_i_ratio) * denoised
            D_i = model(u, sigma_s * s_in, **extra_args)
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * D_i
            # print("sigma_i", sigmas[i], "sigma_ip1", sigmas[i+1],"sigma_down", sigma_down, "sigma_down_i_ratio", sigma_down_i_ratio, "sigma_s_i_ratio", sigma_s_i_ratio, "renoise_coeff", renoise_coeff)
        # Noise addition
        if sigmas[i + 1] > 0 and eta > 0:
            x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
        # logged_x = torch.cat((logged_x, x.unsqueeze(0)), dim=0)
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp_intern(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=None, s_noise=None, noise_sampler=None):
    if hasattr(model, 'model_sampling') and isinstance(model.model_sampling, CONST):
        return sample_dpmpp_2s_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    eta = modules.shared.opts.dpmpp_2s_ancestral_intern_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_2s_ancestral_intern_s_noise if s_noise is None else s_noise
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    s = sigmas[0]
    small_x = nn.functional.interpolate(x, scale_factor=0.5, mode='area')
    den = model(small_x, s * s_in, **extra_args)
    den = nn.functional.interpolate(den, scale_factor=2, mode='area')
    ups_temp = nn.functional.interpolate(temp[0], scale_factor=2, mode='area')
    sigma_down, sigma_up = get_ancestral_step(s, sigmas[1], eta=eta)
    t, t_next = t_fn(s), t_fn(sigma_down)
    r = 1 / 2
    h = t_next - t
    s_ = t + r * h
    x_2 = (sigma_fn(s_) / sigma_fn(t)) * (x + (den - ups_temp)) - (-h * r).expm1() * den
    denoised_2 = model(x_2, sigma_fn(s_) * s_in, **extra_args)
    x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (den - temp[0])) - (-h).expm1() * denoised_2
    large_denoised = x
    x = x + noise_sampler(sigmas[0], sigmas[1]) * s_noise * sigma_up
    sigmas = sigmas[1:] # remove the first sigma we used
    for i in trange(len(sigmas) - 2, disable=disable):
        if sigma_down != 0:
            down_x = nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            denoised = model(down_x, sigmas[i] * s_in, **extra_args)
        else:
            denoised = model(x, sigmas[i] * s_in, **extra_args)
        # denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            mergefactor = min(math.sqrt(i/(len(sigmas) - 2)), 1) 
            print(mergefactor)
            #merge up_den with x
            if mergefactor == 1:
                up_den = large_denoised
                up_temp = nn.functional.interpolate(temp[0], scale_factor=2, mode='area')
                x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (up_den - up_temp)) - (-h * r).expm1() * up_den
            else:
                up_den = nn.functional.interpolate(denoised, scale_factor=2, mode='area')
                print(up_den.max(), large_denoised.max())
                up_den = (up_den * (1-mergefactor)) + (large_denoised * mergefactor)
                print(up_den.max(), large_denoised.max())
                up_temp = nn.functional.interpolate(temp[0], scale_factor=2, mode='area')
                x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (up_den - up_temp)) - (-h * r).expm1() * up_den
            
            
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (up_den - temp[0])) - (-h).expm1() * denoised_2
            large_denoised = denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_sde_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """DPM-Solver++ (stochastic) with CFG++."""
    eta = modules.shared.opts.dpmpp_sde_cfg_pp_eta
    s_noise = modules.shared.opts.dpmpp_sde_cfg_pp_s_noise
    r = modules.shared.opts.dpmpp_sde_cfg_pp_r
    
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]
    
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            dt = sigmas[i + 1] - sigmas[i]
            x = denoised + d * sigmas[i + 1]
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * temp[0] + fac * temp[0]  # Use temp[0] instead of denoised
            x = denoised_2 + to_d(x, sigmas[i], denoised_d) * sd
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x

@torch.no_grad()
def sample_dpmpp_2m_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    t_fn = lambda sigma: sigma.log().neg()

    old_uncond_denoised = None
    uncond_denoised = None
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]
    
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            denoised_mix = -torch.exp(-h) * uncond_denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_uncond_denoised)
        x = denoised + denoised_mix + torch.exp(-h) * x
        old_uncond_denoised = uncond_denoised
    return x

@torch.no_grad()
def sample_ode(model, x, sigmas, extra_args=None, callback=None, disable=None, solver="dopri5", rtol=1e-3, atol=1e-4, max_steps=250):
    """Implements ODE-based sampling."""
    sampler = ODESampler(solver, rtol, atol, max_steps)
    return sampler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

@torch.no_grad()
def sample_dpmpp_3m_sde_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=None, s_noise=None, noise_sampler=None):
    """DPM-Solver++(3M) SDE."""
    eta = modules.shared.opts.dpmpp_3m_sde_cfg_pp_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_3m_sde_cfg_pp_s_noise if s_noise is None else s_noise

    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * (x + (denoised - temp[0])) + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

@torch.no_grad()
def sample_dpmpp_2m_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """DPM-Solver++(2M) with dynamic thresholding."""
    s_noise = modules.shared.opts.dpmpp_2m_dy_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.dpmpp_2m_dy_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.dpmpp_2m_dy_s_extra_steps if s_extra_steps is None else s_extra_steps
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = 2**0.5 - 1
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        t, t_next = t_fn(sigma_hat), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=None,
    s_noise=None,
    noise_sampler=None,
    solver_type=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """DPM-Solver++(2M) SDE with dynamic thresholding."""
    eta = modules.shared.opts.dpmpp_2m_sde_dy_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_2m_sde_dy_s_noise if s_noise is None else s_noise
    solver_type = modules.shared.opts.dpmpp_2m_sde_dy_solver_type if solver_type is None else solver_type
    s_dy_pow = modules.shared.opts.dpmpp_2m_sde_dy_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.dpmpp_2m_sde_dy_s_extra_steps if s_extra_steps is None else s_extra_steps
    if len(sigmas) <= 1:
        return x

    if solver_type not in {"heun", "midpoint"}:
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    gamma = 2**0.5 - 1

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max() * (gamma + 1)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigma_hat.log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigma_hat * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            # TODO not working properly
            if eta:
                x = x + noise_sampler(sigma_hat, sigmas[i + 1] * (gamma + 1)) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=None,
    s_noise=None,
    noise_sampler=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """DPM-Solver++(3M) SDE with dynamic thresholding."""
    eta = modules.shared.opts.dpmpp_3m_sde_dy_eta if eta is None else eta
    s_noise = modules.shared.opts.dpmpp_3m_sde_dy_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.dpmpp_3m_sde_dy_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.dpmpp_3m_sde_dy_s_extra_steps if s_extra_steps is None else s_extra_steps

    if len(sigmas) <= 1:
        return x

    gamma = 2**0.5 - 1

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max() * (gamma + 1)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigma_hat.log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            # TODO not working properly
            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1] * (gamma + 1)) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


@torch.no_grad()
def sample_dpmpp_3m_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=None,
    noise_sampler=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    s_noise = modules.shared.opts.dpmpp_3m_dy_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.dpmpp_3m_dy_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.dpmpp_3m_dy_s_extra_steps if s_extra_steps is None else s_extra_steps
    return sample_dpmpp_3m_sde_dy(
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        0.0,
        s_noise,
        noise_sampler,
        s_dy_pow,
        s_extra_steps,
    )

@torch.no_grad()
def dy_sampling_step_cfg_pp(x, model, sigma_next, i, sigma, sigma_hat, callback, **extra_args):
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    original_shape = x.shape
    batch_size, channels, m, n = original_shape[0], original_shape[1], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, channels, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, channels, m, n)

    with Rescaler(model, c, "nearest-exact", **extra_args) as rescaler:
        denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **rescaler.extra_args)
    if callback is not None:
        callback({"x": c, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})

    d = to_d(c, sigma_hat, temp[0])
    c = denoised + d * sigma_next

    d_list = c.view(batch_size, channels, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = a_list.view(batch_size, channels, m, n, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(batch_size, channels, 2 * m, 2 * n)

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, : 2 * m, : 2 * n] = x
        if extra_row:
            x_expanded[:, :, -1:, : 2 * n + 1] = extra_row_content
        if extra_col:
            x_expanded[:, :, : 2 * m, -1:] = extra_col_content
        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]
        x = x_expanded

    return x


@torch.no_grad()
def sample_euler_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=None,
    s_tmin=None,
    s_tmax=float("inf"),
    s_noise=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """Euler with dynamic thresholding and CFG++."""
    s_churn = modules.shared.opts.euler_dy_cfg_pp_s_churn if s_churn is None else s_churn
    s_tmin = modules.shared.opts.euler_dy_cfg_pp_s_tmin if s_tmin is None else s_tmin
    s_noise = modules.shared.opts.euler_dy_cfg_pp_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.euler_dy_cfg_pp_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.euler_dy_cfg_pp_s_extra_steps if s_extra_steps is None else s_extra_steps
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        # print(sigma_hat)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        d = to_d(x, sigma_hat, temp[0])
        # Euler method
        x = denoised + d * sigmas[i + 1]
        if sigmas[i + 1] > 0 and s_extra_steps:
            if i // 2 == 1:
                x = dy_sampling_step_cfg_pp(x, model, sigmas[i + 1], i, sigmas[i], sigma_hat, callback, **extra_args)
    return x


@torch.no_grad()
def smea_sampling_step_cfg_pp(x, model, sigma_next, i, sigma, sigma_hat, callback, **extra_args):
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    m, n = x.shape[2], x.shape[3]
    x = torch.nn.functional.interpolate(input=x, scale_factor=(1.25, 1.25), mode="nearest-exact")

    with Rescaler(model, x, "nearest-exact", **extra_args) as rescaler:
        denoised = model(x, sigma_hat * x.new_ones([x.shape[0]]), **rescaler.extra_args)
    if callback is not None:
        callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})

    d = to_d(x, sigma_hat, temp[0])
    x = denoised + d * sigma_next
    x = torch.nn.functional.interpolate(input=x, size=(m, n), mode="nearest-exact")
    return x


@torch.no_grad()
def sample_euler_smea_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=None,
    s_tmin=None,
    s_tmax=float("inf"),
    s_noise=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """Euler with SMEA, dynamic thresholding and CFG++."""
    s_churn = modules.shared.opts.euler_smea_dy_cfg_pp_s_churn if s_churn is None else s_churn
    s_tmin = modules.shared.opts.euler_smea_dy_cfg_pp_s_tmin if s_tmin is None else s_tmin
    s_noise = modules.shared.opts.euler_smea_dy_cfg_pp_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.euler_smea_dy_cfg_pp_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.euler_smea_dy_cfg_pp_s_extra_steps if s_extra_steps is None else s_extra_steps
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        d = to_d(x, sigma_hat, temp[0])
        # Euler method
        x = denoised + d * sigmas[i + 1]
        if sigmas[i + 1] > 0 and s_extra_steps:
            if i + 1 // 2 == 1:
                x = dy_sampling_step_cfg_pp(x, model, sigmas[i + 1], i, sigmas[i], sigma_hat, callback, **extra_args)
            if i + 1 // 2 == 0:
                x = smea_sampling_step_cfg_pp(x, model, sigmas[i + 1], i, sigmas[i], sigma_hat, callback, **extra_args)
    return x


@torch.no_grad()
def sample_euler_ancestral_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=None,
    s_noise=None,
    noise_sampler=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """Euler ancestral with dynamic thresholding and CFG++."""
    eta = modules.shared.opts.euler_ancestral_dy_cfg_pp_eta if eta is None else eta
    s_noise = modules.shared.opts.euler_ancestral_dy_cfg_pp_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.euler_ancestral_dy_cfg_pp_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.euler_ancestral_dy_cfg_pp_s_extra_steps if s_extra_steps is None else s_extra_steps
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = 2**0.5 - 1
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigma_hat, sigmas[i + 1], eta=eta)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        d = to_d(x, sigma_hat, temp[0])
        # Euler method
        dt = sigma_down - sigma_hat
        x = denoised + d * sigma_down
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigma_hat, sigmas[i + 1] * (gamma + 1)) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_2m_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=None,
    s_dy_pow=None,
    s_extra_steps=None,
):
    """DPM-Solver++(2M) with dynamic thresholding and CFG++."""
    s_noise = modules.shared.opts.dpmpp_2m_dy_cfg_pp_s_noise if s_noise is None else s_noise
    s_dy_pow = modules.shared.opts.dpmpp_2m_dy_cfg_pp_s_dy_pow if s_dy_pow is None else s_dy_pow
    s_extra_steps = modules.shared.opts.dpmpp_2m_dy_cfg_pp_s_extra_steps if s_extra_steps is None else s_extra_steps    
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    t_fn = lambda sigma: sigma.log().neg()

    old_uncond_denoised = None
    uncond_denoised = None
    h_last = None
    h = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = 2**0.5 - 1
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        t, t_next = t_fn(sigma_hat), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            denoised_mix = -torch.exp(-h) * uncond_denoised
        else:
            r = h_last / h
            denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_uncond_denoised)
        x = denoised + denoised_mix + torch.exp(-h) * x
        old_uncond_denoised = uncond_denoised
        h_last = h
    return x

@torch.no_grad()
def sample_clyb_4m_sde_momentumized(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1., noise_sampler=None, momentum=0.0):
    """DPM-Solver++(3M) SDE, modified with an extra SDE, and momentumized in both the SDE and ODE(?). 'its a first' - Clybius 2023
    The expression for d1 is derived from the extrapolation formula given in the paper “Diffusion Monte Carlo with stochastic Hamiltonians” by M. Foulkes, L. Mitas, R. Needs, and G. Rajagopal. The formula is given as follows:
    d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
    (if this is an incorrect citing, we blame Google's Bard and OpenAI's ChatGPT for this and NOT me :^) )

    where d1_0, d1_1, and d1_2 are defined as follows:
    d1_0 = (denoised - denoised_1) / r2
    d1_1 = (denoised_1 - denoised_2) / r1
    d1_2 = (denoised_2 - denoised_3) / r0

    The variables r0, r1, and r2 are defined as follows:
    r0 = h_3 / h_2
    r1 = h_2 / h
    r2 = h / h_1
    """

    def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
        return momentum_vel

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2, denoised_3 = None, None, None
    h_1, h_2, h_3 = None, None, None
    vel, vel_sde = None, None
    for i in trange(len(sigmas) - 1, disable=disable):
        time = sigmas[i] / sigma_max
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)
            x_diff = momentum_func((-h_eta).expm1().neg() * denoised, vel, time)
            vel = x_diff
            x = torch.exp(-h_eta) * x + vel

            if h_3 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                r2 = h_3 / h
                d1_0 = (denoised   - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1_2 = (denoised_2 - denoised_3) / r2
                # d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1) + ((d1_0 - d1_1) * r2 / (r1 + r2) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r1 + r2) * (r0 + r1))
                # d2 = (d1_0 - d1_1) / (r0 + r1) + ((d1_0 - d1_1) * r2 / (r1 + r2) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r1 + r2) * (r0 + r1))

                # r0 = h_3 / h_2
                # r1 = h_2 / h
                # r2 = h / h_1
                # d1_0 = (denoised - denoised_1) / r2
                # d1_1 = (denoised_1 - denoised_2) / r1
                # d1_2 = (denoised_2 - denoised_3) / r0
                d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
                d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
                phi_3 = h_eta.neg().expm1() / h_eta + 1
                phi_4 = phi_3 / h_eta - 0.5
                sde_diff = momentum_func(phi_3 * d1 - phi_4 * d2, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde
            elif h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                sde_diff = momentum_func(phi_2 * d1 - phi_3 * d2, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                sde_diff = momentum_func(phi_2 * d, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

            denoised_1, denoised_2, denoised_3 = denoised, denoised_1, denoised_2
            h_1, h_2, h_3 = h, h_1, h_2

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

    return 

class DenoiserModel(Protocol):
  def __call__(self, x: FloatTensor, t: FloatTensor, *args, **kwargs) -> FloatTensor: ...

class RefinedExpCallbackPayload(TypedDict):
  x: FloatTensor
  i: int
  sigma: FloatTensor
  sigma_hat: FloatTensor

class RefinedExpCallback(Protocol):
  def __call__(self, payload: RefinedExpCallbackPayload) -> None: ...

class NoiseSampler(Protocol):
  def __call__(self, x: FloatTensor) -> FloatTensor: ...

class StepOutput(NamedTuple):
  x_next: FloatTensor
  denoised: FloatTensor
  denoised2: FloatTensor
  vel: FloatTensor
  vel_2: FloatTensor

def _gamma(
  n: int,
) -> int:
  """
  https://en.wikipedia.org/wiki/Gamma_function
  for every positive integer n,
  Γ(n) = (n-1)!
  """
  return math.factorial(n-1)

def _incomplete_gamma(
  s: int,
  x: float,
  gamma_s: Optional[int] = None
) -> float:
  """
  https://en.wikipedia.org/wiki/Incomplete_gamma_function#Special_values
  if s is a positive integer,
  Γ(s, x) = (s-1)!*∑{k=0..s-1}(x^k/k!)
  """
  if gamma_s is None:
    gamma_s = _gamma(s)

  sum_: float = 0
  # {k=0..s-1} inclusive
  for k in range(s):
    numerator: float = x**k
    denom: int = math.factorial(k)
    quotient: float = numerator/denom
    sum_ += quotient
  incomplete_gamma_: float = sum_ * math.exp(-x) * gamma_s
  return incomplete_gamma_

# by Katherine Crowson
def _phi_1(neg_h: FloatTensor):
  return torch.nan_to_num(torch.expm1(neg_h) / neg_h, nan=1.0)

# by Katherine Crowson
def _phi_2(neg_h: FloatTensor):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h) / neg_h**2, nan=0.5)

# by Katherine Crowson
def _phi_3(neg_h: FloatTensor):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h - neg_h**2 / 2) / neg_h**3, nan=1 / 6)

def _phi(
  neg_h: float,
  j: int,
):
  """
  For j={1,2,3}: you could alternatively use Kat's phi_1, phi_2, phi_3 which perform fewer steps

  Lemma 1
  https://arxiv.org/abs/2308.02157
  ϕj(-h) = 1/h^j*∫{0..h}(e^(τ-h)*(τ^(j-1))/((j-1)!)dτ)

  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84
  = 1/h^j*[(e^(-h)*(-τ)^(-j)*τ(j))/((j-1)!)]{0..h}
  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84+between+0+and+h
  = 1/h^j*((e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h)))/(j-1)!)
  = (e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h))/((j-1)!*h^j)
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/(j-1)!
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/Γ(j)
  = (e^(-h)*(-h)^(-j)*(1-Γ(j,-h)/Γ(j))

  requires j>0
  """
  assert j > 0
  gamma_: float = _gamma(j)
  incomp_gamma_: float = _incomplete_gamma(j, neg_h, gamma_s=gamma_)

  phi_: float = math.exp(neg_h) * neg_h**-j * (1-incomp_gamma_/gamma_)

  return phi_

class RESDECoeffsSecondOrder(NamedTuple):
  a2_1: float
  b1: float
  b2: float

def _de_second_order(
  h: float,
  c2: float,
  simple_phi_calc = False,
) -> RESDECoeffsSecondOrder:
  """
  Table 3
  https://arxiv.org/abs/2308.02157
  ϕi,j := ϕi,j(-h) = ϕi(-cj*h)
  a2_1 = c2ϕ1,2
       = c2ϕ1(-c2*h)
  b1 = ϕ1 - ϕ2/c2
  """
  if simple_phi_calc:
    # Kat computed simpler expressions for phi for cases j={1,2,3}
    a2_1: float = c2 * _phi_1(-c2*h)
    phi1: float = _phi_1(-h)
    phi2: float = _phi_2(-h)
  else:
    # I computed general solution instead.
    # they're close, but there are slight differences. not sure which would be more prone to numerical error.
    a2_1: float = c2 * _phi(j=1, neg_h=-c2*h)
    phi1: float = _phi(j=1, neg_h=-h)
    phi2: float = _phi(j=2, neg_h=-h)
  phi2_c2: float = phi2/c2
  b1: float = phi1 - phi2_c2
  b2: float = phi2_c2
  return RESDECoeffsSecondOrder(
    a2_1=a2_1,
    b1=b1,
    b2=b2,
  )  

def _refined_exp_sosu_step(
  model: DenoiserModel,
  x: FloatTensor,
  sigma: FloatTensor,
  sigma_next: FloatTensor,
  c2 = 0.5,
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  simple_phi_calc = False,
  momentum = 0.0,
  vel = None,
  vel_2 = None,
  time = None
) -> StepOutput:
  """
  Algorithm 1 "RES Second order Single Update Step with c2"
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigma (`FloatTensor`): timestep to denoise
    sigma_next (`FloatTensor`): timestep+1 to denoise
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    pbar (`tqdm`, *optional*, defaults to `None`): progress bar to update after each model call
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences.
  """

  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  lam_next, lam = (s.log().neg() for s in (sigma_next, sigma))

  # type hints aren't strictly true regarding float vs FloatTensor.
  # everything gets promoted to `FloatTensor` after interacting with `sigma: FloatTensor`.
  # I will use float to indicate any variables which are scalars.
  h: float = lam_next - lam
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  
  denoised: FloatTensor = model(x, sigma.repeat(x.size(0)), **extra_args)
  # if pbar is not None:
    # pbar.update(0.5)

  c2_h: float = c2*h

  diff_2 = momentum_func(a2_1*h*denoised, vel_2, time)
  vel_2 = diff_2
  x_2: FloatTensor = math.exp(-c2_h)*x + diff_2
  lam_2: float = lam + c2_h
  sigma_2: float = lam_2.neg().exp()

  denoised2: FloatTensor = model(x_2, sigma_2.repeat(x_2.size(0)), **extra_args)
  if pbar is not None:
    pbar.update()

  diff = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)
  vel = diff

  x_next: FloatTensor = math.exp(-h)*x + diff
  
  return StepOutput(
    x_next=x_next,
    denoised=denoised,
    denoised2=denoised2,
    vel=vel,
    vel_2=vel_2,
  )
  

@no_grad()
def sample_refined_exp_s(
  model: FloatTensor,
  x: FloatTensor,
  sigmas: FloatTensor,
  denoise_to_zero: bool = True,
  extra_args: Dict[str, Any] = {},
  callback: Optional[RefinedExpCallback] = None,
  disable: Optional[bool] = None,
  ita: FloatTensor = torch.zeros((1,)),
  c2 = .5,
  noise_sampler: NoiseSampler = torch.randn_like,
  simple_phi_calc = False,
  momentum = 0.0,
):
  """
  Refined Exponential Solver (S).
  Algorithm 2 "RES Single-Step Sampler" with Algorithm 1 second-order step
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigmas (`FloatTensor`): sigmas (ideally an exponential schedule!) e.g. get_sigmas_exponential(n=25, sigma_min=model.sigma_min, sigma_max=model.sigma_max)
    denoise_to_zero (`bool`, *optional*, defaults to `True`): whether to finish with a first-order step down to 0 (rather than stopping at sigma_min). True = fully denoise image. False = match Algorithm 2 in paper
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    callback (`RefinedExpCallback`, *optional*, defaults to `None`): you can supply this callback to see the intermediate denoising results, e.g. to preview each step of the denoising process
    disable (`bool`, *optional*, defaults to `False`): whether to hide `tqdm`'s progress bar animation from being printed
    ita (`FloatTensor`, *optional*, defaults to 0.): degree of stochasticity, η, for each timestep. tensor shape must be broadcastable to 1-dimensional tensor with length `len(sigmas) if denoise_to_zero else len(sigmas)-1`. each element should be from 0 to 1.
         - if used: batch noise doesn't match non-batch
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    noise_sampler (`NoiseSampler`, *optional*, defaults to `torch.randn_like`): method used for adding noise
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences.
  """
  #assert sigmas[-1] == 0
  device = x.device
  ita = ita.to(device)
  sigmas = sigmas.to(device)

  sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

  vel, vel_2 = None, None
  with tqdm(disable=disable, total=len(sigmas)-(1 if denoise_to_zero else 2)) as pbar:
    for i, (sigma, sigma_next) in enumerate(pairwise(sigmas[:-1].split(1))):
      time = sigmas[i] / sigma_max
      if 'sigma' not in locals():
        sigma = sigmas[i]
      eps = torch.randn_like(x).float()
      sigma_hat = sigma * (1 + ita)
      x_hat = x + (sigma_hat ** 2 - sigma ** 2).sqrt() * eps
      x_next, denoised, denoised2, vel, vel_2 = _refined_exp_sosu_step(
        model,
        x_hat,
        sigma_hat,
        sigma_next,
        c2=c2,
        extra_args=extra_args,
        pbar=pbar,
        simple_phi_calc=simple_phi_calc,
        momentum = momentum,
        vel = vel,
        vel_2 = vel_2,
        time = time
      )
      if callback is not None:
        payload = RefinedExpCallbackPayload(
          x=x,
          i=i,
          sigma=sigma,
          sigma_hat=sigma_hat,
          denoised=denoised,
          denoised2=denoised2,
        )
        callback(payload)
      x = x_next
    if denoise_to_zero:
      eps = torch.randn_like(x).float()
      sigma_hat = sigma * (1 + ita)
      x_hat = x + (sigma_hat ** 2 - sigma ** 2).sqrt() * eps
      x_next: FloatTensor = model(x_hat, sigma.to(x_hat.device).repeat(x_hat.size(0)), **extra_args)
      pbar.update()

      if callback is not None:
        payload = RefinedExpCallbackPayload(
          x=x,
          i=i,
          sigma=sigma,
          sigma_hat=sigma_hat,
          denoised=denoised,
          denoised2=denoised2,
        )
        callback(payload)


      x = x_next
  return x

# Many thanks to Kat + Birch-San for this wonderful sampler implementation! https://github.com/Birch-san/sdxl-play/commits/res/
def sample_res_solver(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_sampler=None, denoise_to_zero=True, simple_phi_calc=False, c2=0.5, ita=torch.Tensor((0.0,)), momentum=0.0):
    return sample_refined_exp_s(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, noise_sampler=noise_sampler, denoise_to_zero=denoise_to_zero, simple_phi_calc=simple_phi_calc, c2=c2, ita=ita, momentum=momentum)

@torch.no_grad()
def sample_kohaku_lonyu_yog_cfg_pp(
    model, 
    x, 
    sigmas, 
    extra_args=None, 
    callback=None, 
    disable=None, 
    s_churn=None, 
    s_tmin=None,
    s_tmax=float('inf'), 
    s_noise=None, 
    noise_sampler=None, 
    eta=None
):
    """Kohaku_LoNyu_Yog sampler with CFG++ implementation"""
    # Get values from shared options if not provided
    s_churn = modules.shared.opts.kohaku_lonyu_yog_s_cfgpp_churn if s_churn is None else s_churn
    s_tmin = modules.shared.opts.kohaku_lonyu_yog_s_cfgpp_tmin if s_tmin is None else s_tmin
    s_noise = modules.shared.opts.kohaku_lonyu_yog_s_cfgpp_noise if s_noise is None else s_noise
    eta = modules.shared.opts.kohaku_lonyu_yog_cfgpp_eta if eta is None else eta

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    
    # Set up CFG++ components
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]
    
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ldm_patched.modules.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )
    
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, temp[0])  # Use uncond_denoised from CFG++
        
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            
        dt = sigma_down - sigmas[i]
        
        if i <= (len(sigmas) - 1) / 2:
            x2 = -x
            denoised2 = model(x2, sigma_hat * s_in, **extra_args)
            d2 = to_d(x2, sigma_hat, temp[0])  # Use uncond_denoised from CFG++
            x3 = x + ((d + d2) / 2) * dt
            denoised3 = model(x3, sigma_hat * s_in, **extra_args)
            d3 = to_d(x3, sigma_hat, temp[0])  # Use uncond_denoised from CFG++
            real_d = (d + d3) / 2
            x = x + real_d * dt
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        else:
            x = x + d * dt
            
    return x

def sample_custom(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Custom sampler that uses configurations from shared options"""
    
    # Get sampler parameters from shared options
    sampler_name = modules.shared.opts.custom_sampler_name
    eta = modules.shared.opts.custom_sampler_eta
    s_noise = modules.shared.opts.custom_sampler_s_noise
    solver_type = modules.shared.opts.custom_sampler_solver_type
    r = modules.shared.opts.custom_sampler_r
    cfg_scale = modules.shared.opts.custom_cfg_conds
    cfg_scale2 = modules.shared.opts.custom_cfg_cond2_negative

    # Get the appropriate sampler function
    sampler_functions = {
            'euler_comfy': sample_euler,
            'euler_ancestral_comfy': sample_euler_ancestral,
            'heun_comfy': sample_heun,
            'dpmpp_2s_ancestral_comfy': sample_dpmpp_2s_ancestral,
            'dpmpp_sde_comfy': sample_dpmpp_sde,
            'dpmpp_2m_comfy': sample_dpmpp_2m,
            'dpmpp_2m_sde_comfy': sample_dpmpp_2m_sde,
            'dpmpp_3m_sde_comfy': sample_dpmpp_3m_sde,
            'euler_ancestral_turbo': sample_euler_ancestral,
            'dpmpp_2m_turbo': sample_dpmpp_2m,
            'dpmpp_2m_sde_turbo': sample_dpmpp_2m_sde,
            'ddpm': sample_ddpm,
            'heunpp2': sample_heunpp2,
            'ipndm': sample_ipndm,
            'ipndm_v': sample_ipndm_v,
            'deis': sample_deis,
            'euler_cfg_pp': sample_euler_cfg_pp,
            'euler_ancestral_cfg_pp': sample_euler_ancestral_cfg_pp,
            'sample_euler_ancestral_RF': sample_euler_ancestral_RF,
            'dpmpp_2s_ancestral_cfg_pp': sample_dpmpp_2s_ancestral_cfg_pp,
            'sample_dpmpp_2s_ancestral_RF': sample_dpmpp_2s_ancestral_RF,
            'dpmpp_2s_ancestral_cfg_pp_dyn': sample_dpmpp_2s_ancestral_cfg_pp_dyn,
            'dpmpp_2s_ancestral_cfg_pp_intern': sample_dpmpp_2s_ancestral_cfg_pp_intern,
            'dpmpp_sde_cfg_pp': sample_dpmpp_sde_cfg_pp,
            'dpmpp_2m_cfg_pp': sample_dpmpp_2m_cfg_pp,
            'dpmpp_3m_sde_cfg_pp': sample_dpmpp_3m_sde_cfg_pp,
            'dpmpp_2m_dy': sample_dpmpp_2m_dy,
            'dpmpp_3m_dy': sample_dpmpp_3m_dy,
            'dpmpp_3m_sde_dy': sample_dpmpp_3m_sde_dy,
            'euler_dy_cfg_pp': sample_euler_dy_cfg_pp,
            'euler_smea_dy_cfg_pp': sample_euler_smea_dy_cfg_pp,
            'euler_ancestral_dy_cfg_pp': sample_euler_ancestral_dy_cfg_pp,
            'dpmpp_2m_dy_cfg_pp': sample_dpmpp_2m_dy_cfg_pp,
            'clyb_4m_sde_momentumized': sample_clyb_4m_sde_momentumized,
            'res_solver': sample_res_solver,
            'kohaku_lonyu_yog_cfg_pp': sample_kohaku_lonyu_yog_cfg_pp,
        }

    sampler_function = sampler_functions.get(sampler_name)
    if sampler_function is None:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    # Prepare sampler kwargs based on which sampler is selected
    kwargs = {
        "model": model,
        "x": x,
        "sigmas": sigmas,
        "extra_args": extra_args,
        "callback": callback,
        "disable": disable,
    }

    # Add additional parameters based on sampler type
    if "cfg" in sampler_name:
        kwargs["cfg_scale"] = cfg_scale
    if "sde" in sampler_name:
        kwargs.update({
            "eta": eta,
            "s_noise": s_noise,
        })
    if "2m_sde" in sampler_name:
        kwargs["solver_type"] = solver_type
    if any(x in sampler_name for x in ["sde", "dpmpp"]):
        kwargs["r"] = r

    # Call the sampler
    return sampler_function(**kwargs)