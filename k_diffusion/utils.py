
import hashlib
import math
import random
import shutil
import threading
import time
import urllib
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union
)

import numpy as np  # for optional usage in temp_seed, if available
import safetensors
import torch
from PIL import Image
from torch import nn, optim
from torch.utils import data
from torchvision.transforms import functional as TF


#
# ====================
#  Context Managers
# ====================
#

@contextmanager
def train_mode(model: nn.Module, mode: bool = True) -> Generator[nn.Module, None, None]:
    """
    A context manager that places a model into training mode and
    restores the previous mode on exit.
    """
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model: nn.Module) -> Generator[nn.Module, None, None]:
    """
    A context manager that places a model into evaluation mode and
    restores the previous mode on exit.
    """
    return train_mode(model, False)


@contextmanager
def tf32_mode(cudnn: Optional[bool] = None, matmul: Optional[bool] = None) -> Generator[None, None, None]:
    """
    A context manager that sets whether TF32 is allowed on cuDNN or matmul.
    """
    cudnn_old = torch.backends.cudnn.allow_tf32
    matmul_old = torch.backends.cuda.matmul.allow_tf32
    try:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_old
        torch.backends.cuda.matmul.allow_tf32 = matmul_old


@contextmanager
def timeit(name: str = "Block") -> Generator[None, None, None]:
    """
    A simple context manager to measure execution time of a code block.
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[{name}] Elapsed time: {elapsed:.6f} seconds.")


@contextmanager
def temp_seed(seed: Optional[int] = None) -> Generator[None, None, None]:
    """
    A context manager that temporarily sets the RNG seed for torch, random, and numpy
    (if installed). Restores the previous state upon exit.
    """
    if seed is None:
        yield
        return

    # Store old states
    old_state_torch = torch.random.get_rng_state()
    old_state_python = random.getstate()
    old_state_numpy = None

    # Try to store NumPy’s state if available
    try:
        old_state_numpy = np.random.get_state()
    except NameError:
        pass

    # Set new seeds
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        np.random.seed(seed)
    except NameError:
        pass

    try:
        yield
    finally:
        # Restore old states
        torch.random.set_rng_state(old_state_torch)
        random.setstate(old_state_python)
        if old_state_numpy is not None:
            np.random.set_state(old_state_numpy)


#
# =========================
#  PIL / Tensor Utilities
# =========================
#

def from_pil_image(x: Image.Image) -> torch.Tensor:
    """
    Converts from a PIL image to a normalized [-1, 1] tensor.
    If the image is grayscale, we add an extra channel.
    """
    tensor = TF.to_tensor(x)
    if tensor.ndim == 2:
        tensor = tensor[..., None]
    return tensor * 2 - 1


def to_pil_image(x: torch.Tensor) -> Image.Image:
    """
    Converts from a normalized [-1, 1] tensor back to a PIL image.
    If x.ndim == 4, we assume a batch of size 1.
    """
    if x.ndim == 4:
        assert x.shape[0] == 1, "Batch dimension should be 1 for PIL conversion."
        x = x[0]
    if x.shape[0] == 1:
        x = x[0]
    return TF.to_pil_image((x.clamp(-1, 1) + 1) / 2)


def hf_datasets_augs_helper(
    examples: Dict[str, List[Image.Image]],
    transform: Callable[[Image.Image], torch.Tensor],
    image_key: str,
    mode: str = "RGB"
) -> Dict[str, List[torch.Tensor]]:
    """
    Apply a transform for HuggingFace Datasets, converting images under `image_key`
    to a specified mode, then to a tensor.
    """
    images = [transform(image.convert(mode)) for image in examples[image_key]]
    return {image_key: images}


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """
    Appends dimensions to the end of a tensor until it has `target_dims` dimensions.
    Raises ValueError if x already has more than `target_dims` dims.
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"Input has {x.ndim} dims but target_dims is {target_dims}, which is less.")
    return x[(...,) + (None,) * dims_to_append]


def n_params(module: nn.Module) -> int:
    """
    Returns the number of trainable parameters in a given nn.Module.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


#
# ================================
#  File & Downloading Utilities
# ================================
#

def download_file(path: Union[str, Path], url: str, digest: Optional[str] = None) -> Path:
    """
    Downloads a file if it does not exist, optionally checking its SHA-256 hash.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url) as response, open(path, 'wb') as f:
            shutil.copyfileobj(response, f)
    if digest is not None:
        file_digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        if digest != file_digest:
            raise OSError(f"Hash of {path} (url: {url}) failed to validate")
    return path


def get_safetensors_metadata(path: Union[str, Path]) -> Dict[str, str]:
    """
    Retrieves the metadata from a safetensors file.
    """
    path = str(path)
    return safetensors.safe_open(path, "pt").metadata()


#
# ===============
#  EMA Utilities
# ===============
#

@torch.no_grad()
def ema_update(model: nn.Module, averaged_model: nn.Module, decay: float) -> None:
    """
    Incorporates updated model parameters into an exponential moving averaged
    version of a model. Call after each optimizer step.
    """
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys(), "Parameter sets do not match."

    for name, param in model_params.items():
        averaged_params[name].lerp_(param, 1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys(), "Buffer sets do not match."

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class EMAWarmup:
    """
    Implements an EMA warmup using an inverse decay schedule.

    If inv_gamma=1 and power=1, implements a simple average.
    inv_gamma=1, power=2/3 are good values for training long runs (~1M steps).
    """
    def __init__(
        self,
        inv_gamma: float = 1.0,
        power: float = 1.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        start_at: int = 0,
        last_epoch: int = 0
    ):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the class as a dictionary.
        """
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the class's state.
        """
        self.__dict__.update(state_dict)

    def get_value(self) -> float:
        """
        Gets the current EMA decay rate.
        """
        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return 0.0 if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self) -> None:
        """
        Updates the step count (last_epoch).
        """
        self.last_epoch += 1


def ema_update_dict(values: Dict[str, float], updates: Dict[str, float], decay: float) -> Dict[str, float]:
    """
    Updates a dictionary of (key -> EMA value) pairs with a new dictionary of updates,
    applying an exponential moving average with the given decay.
    """
    for k, v in updates.items():
        if k not in values:
            values[k] = v
        else:
            values[k] *= decay
            values[k] += (1 - decay) * v
    return values


#
# ========================
#  Learning Rate Schedules
# ========================
#

class InverseLR(optim.lr_scheduler._LRScheduler):
    """
    Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as base_lr.

    `inv_gamma` is the number of steps required for the learning rate to decay
    to (1/2)**power of its original value.
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        inv_gamma: float = 1.0,
        power: float = 1.0,
        warmup: float = 0.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.inv_gamma = inv_gamma
        self.power = power
        if not (0.0 <= warmup < 1.0):
            raise ValueError("Invalid value for warmup.")
        self.warmup = warmup
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> List[float]:
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [
            warmup * max(self.min_lr, base_lr * lr_mult)
            for base_lr in self.base_lrs
        ]


class ExponentialLR(optim.lr_scheduler._LRScheduler):
    """
    Implements an exponential learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as base_lr.

    Decays the LR continuously by `decay` every `num_steps` steps.
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_steps: float,
        decay: float = 0.5,
        warmup: float = 0.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.num_steps = num_steps
        self.decay = decay
        if not (0.0 <= warmup < 1.0):
            raise ValueError("Invalid value for warmup.")
        self.warmup = warmup
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> List[float]:
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (self.decay ** (1 / self.num_steps)) ** self.last_epoch
        return [
            warmup * max(self.min_lr, base_lr * lr_mult)
            for base_lr in self.base_lrs
        ]


class ConstantLRWithWarmup(optim.lr_scheduler._LRScheduler):
    """
    Implements a constant learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as base_lr.
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        if not (0.0 <= warmup < 1.0):
            raise ValueError("Invalid value for warmup.")
        self.warmup = warmup
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> List[float]:
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        return [
            warmup * base_lr
            for base_lr in self.base_lrs
        ]


#
# =========================
#  Stratified Sampling Utils
# =========================
#

stratified_settings = threading.local()

@contextmanager
def enable_stratified(group: int = 0, groups: int = 1, disable: bool = False) -> Generator[None, None, None]:
    """
    A context manager that enables stratified sampling within its scope.
    """
    try:
        stratified_settings.disable = disable
        stratified_settings.group = group
        stratified_settings.groups = groups
        yield
    finally:
        del stratified_settings.disable
        del stratified_settings.group
        del stratified_settings.groups


@contextmanager
def enable_stratified_accelerate(accelerator: Any, disable: bool = False) -> Generator[None, None, None]:
    """
    A context manager that enables stratified sampling, distributing the strata
    across processes and gradient accumulation steps using settings from
    Hugging Face Accelerate.
    """
    try:
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        # If accelerator.gradient_state.num_steps is unavailable, fallback or override:
        acc_steps = getattr(accelerator.gradient_state, "num_steps", 1)
        acc_step = getattr(accelerator, "step", 0) % acc_steps

        group = rank * acc_steps + acc_step
        groups = world_size * acc_steps
        with enable_stratified(group, groups, disable=disable):
            yield
    finally:
        pass


def stratified_uniform(
    shape: torch.Size,
    group: int = 0,
    groups: int = 1,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Draws stratified samples from a uniform distribution in [0, 1].
    """
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")

    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def stratified_with_settings(
    shape: torch.Size,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Draws stratified samples from a uniform distribution using the current
    context manager settings. Falls back to standard uniform if disabled
    or not configured.
    """
    if not hasattr(stratified_settings, 'disable') or stratified_settings.disable:
        return torch.rand(shape, dtype=dtype, device=device)
    return stratified_uniform(
        shape,
        stratified_settings.group,
        stratified_settings.groups,
        dtype=dtype,
        device=device
    )


#
# ==========================
#  Distribution Samplers
# ==========================
#

def rand_log_normal(
    shape: torch.Size,
    loc: float = 0.0,
    scale: float = 1.0,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Draws samples from a log-normal distribution using Normal(loc, scale).
    """
    u = stratified_with_settings(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def rand_log_logistic(
    shape: torch.Size,
    loc: float = 0.0,
    scale: float = 1.0,
    min_value: float = 0.0,
    max_value: float = float('inf'),
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Draws samples from an optionally truncated log-logistic distribution.
    """
    min_value_t = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value_t = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value_t.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value_t.log().sub(loc).div(scale).sigmoid()
    u = stratified_with_settings(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(
    shape: torch.Size,
    min_value: float,
    max_value: float,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Draws samples from a log-uniform distribution.
    """
    min_log = math.log(min_value)
    max_log = math.log(max_value)
    return (stratified_with_settings(shape, device=device, dtype=dtype) * (max_log - min_log) + min_log).exp()


def rand_v_diffusion(
    shape: torch.Size,
    sigma_data: float = 1.0,
    min_value: float = 0.0,
    max_value: float = float('inf'),
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Draws samples from a truncated v-diffusion training timestep distribution.
    """
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = stratified_with_settings(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data


def rand_cosine_interpolated(
    shape: torch.Size,
    image_d: float,
    noise_d_low: float,
    noise_d_high: float,
    sigma_data: float = 1.0,
    min_value: float = 1e-3,
    max_value: float = 1e3,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Draws samples from an interpolated cosine timestep distribution (from Simple Diffusion).
    """
    def logsnr_schedule_cosine(t: torch.Tensor, logsnr_min: float, logsnr_max: float) -> torch.Tensor:
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2.0 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(
        t: torch.Tensor,
        image_d_: float,
        noise_d_: float,
        logsnr_min_: float,
        logsnr_max_: float
    ) -> torch.Tensor:
        shift = 2.0 * math.log(noise_d_ / image_d_)
        return logsnr_schedule_cosine(t, logsnr_min_ - shift, logsnr_max_ - shift) + shift

    def logsnr_schedule_cosine_interpolated(
        t: torch.Tensor,
        image_d_: float,
        noise_d_low_: float,
        noise_d_high_: float,
        logsnr_min_: float,
        logsnr_max_: float
    ) -> torch.Tensor:
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d_, noise_d_low_, logsnr_min_, logsnr_max_)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d_, noise_d_high_, logsnr_min_, logsnr_max_)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2.0 * math.log(min_value / sigma_data)
    logsnr_max = -2.0 * math.log(max_value / sigma_data)
    u = stratified_with_settings(shape, device=device, dtype=dtype)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


def rand_split_log_normal(
    shape: torch.Size,
    loc: float,
    scale_1: float,
    scale_2: float,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Draws samples from a split lognormal distribution.
    """
    n = torch.randn(shape, device=device, dtype=dtype).abs()
    u = torch.rand(shape, device=device, dtype=dtype)
    n_left = n * -scale_1 + loc
    n_right = n * scale_2 + loc
    ratio = scale_1 / (scale_1 + scale_2)
    return torch.where(u < ratio, n_left, n_right).exp()


#
# =======================================
#  New Distributions (Novel Enhancements)
# =======================================
#

def rand_laplace(
    shape: torch.Size,
    loc: float = 0.0,
    scale: float = 1.0,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Draws samples from a Laplace distribution with parameters (loc, scale).
    PDF: p(x) = (1 / (2 * scale)) * exp(-|x - loc| / scale).
    """
    # [-0.5, 0.5] -> scale & loc
    u = stratified_with_settings(shape, dtype=dtype, device=device) - 0.5
    return loc - scale * torch.sign(u) * torch.log1p(-2 * u.abs())


def rand_beta(
    shape: torch.Size,
    alpha: float = 2.0,
    beta: float = 2.0,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Draws samples from a Beta distribution with parameters (alpha, beta).
    Uses the PyTorch Beta distribution if available with stratified sampling for U.
    """
    u = stratified_with_settings(shape, dtype=dtype, device=device)
    return torch.distributions.Beta(alpha, beta).icdf(u)


#
# =====================
#  Dataset & Logging
# =====================
#

class FolderOfImages(data.Dataset):
    """
    Recursively finds all images in a directory. Does not support
    classes or targets by default (returns just (image,)).

    Args:
        root: Path to the directory of images.
        transform: Optional transform to apply to images.
        shuffle: If True, randomizes the order of file paths on initialization.
                 This does not shuffle each epoch; for that, wrap in a DataLoader
                 with shuffle=True.
        seed: Random seed for shuffling the image list consistently if `shuffle=True`.
    """
    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        shuffle: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = nn.Identity() if transform is None else transform

        # Find image paths
        self.paths = sorted(
            path for path in self.root.rglob('*')
            if path.suffix.lower() in self.IMG_EXTENSIONS
        )

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.paths)

    def __repr__(self) -> str:
        return f'FolderOfImages(root="{self.root}", len={len(self)})'

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, key: int) -> Tuple[torch.Tensor]:
        path = self.paths[key]
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        image_tensor = self.transform(image)
        return (image_tensor,)

    def __iter__(self) -> Iterable[Tuple[torch.Tensor]]:
        """
        Optional: allows iteration directly over the dataset.
        Be mindful that PyTorch DataLoader typically calls __getitem__ in parallel.
        """
        for i in range(len(self)):
            yield self[i]


class CSVLogger:
    """
    A simple CSV logger that writes rows to a file. Allows appending or
    creating a new file. Now supports a context manager interface and a read method.
    """
    def __init__(self, filename: Union[str, Path], columns: List[str]):
        self.filename = Path(filename)
        self.columns = columns
        self.file = None

    def __enter__(self) -> "CSVLogger":
        # Open in append mode if file exists, else write mode
        if self.filename.exists():
            self.file = open(self.filename, 'a', encoding='utf-8')
        else:
            self.file = open(self.filename, 'w', encoding='utf-8')
            self.write(*self.columns)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.file is not None:
            self.file.close()

    def write(self, *args: Any) -> None:
        if self.file is None:
            raise ValueError("CSVLogger file is not open. Use `with CSVLogger(...)` or call open() manually.")
        print(*args, sep=',', file=self.file, flush=True)

    def read_rows(self) -> List[List[str]]:
        """
        Reads all rows from the CSV file. For large files, consider using
        an iterator or a more efficient library.
        """
        if not self.filename.exists():
            return []
        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = [line.strip().split(',') for line in f]
        return lines


#
# End of Module
#
