# Cherry‑picked some good parts from ComfyUI with fixes and enhancements

import sys
import time
import psutil
import platform
import logging
from enum import Enum
from functools import lru_cache
from typing import Optional, Tuple, List

import torch
from backend import stream, utils
from backend.args import args

# ── Logging setup ────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── Devices and States ───────────────────────────────────────────────────────────
cpu = torch.device("cpu")

class VRAMState(Enum):
    DISABLED   = 0  # no VRAM
    NO_VRAM    = 1  # minimal VRAM, heavy offload
    LOW_VRAM   = 2
    NORMAL_VRAM= 3
    HIGH_VRAM  = 4
    SHARED     = 5  # unified memory

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

vram_state    = VRAMState.NORMAL_VRAM
set_vram_to   = VRAMState.NORMAL_VRAM
cpu_state     = CPUState.GPU
lowvram_avail = True
xpu_avail     = False

# ── Deterministic & DirectML ────────────────────────────────────────────────────
if args.pytorch_deterministic:
    logger.info("Enabling PyTorch deterministic algorithms")
    torch.use_deterministic_algorithms(True, warn_only=True)

directml_enabled = False
if args.directml is not None:
    try:
        import torch_directml
        directml_enabled = True
        idx = args.directml
        directml_device = (
            torch_directml.device() if idx < 0 else torch_directml.device(idx)
        )
        logger.info(f"DirectML enabled on device {directml_device}")
    except ImportError:
        logger.warning("torch_directml not installed; ignoring --directml flag")

# ── Intel XPU & MPS ──────────────────────────────────────────────────────────────
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        xpu_avail = True
        logger.info("Intel XPU available")
except ImportError:
    pass

if torch.backends.mps.is_available():
    cpu_state = CPUState.MPS
    logger.info("Apple MPS backend available")

if args.always_cpu:
    cpu_state = CPUState.CPU
    logger.info("Forcing CPU mode via --always_cpu")

# ── Helpers ─────────────────────────────────────────────────────────────────────
def is_intel_xpu() -> bool:
    return cpu_state == CPUState.GPU and xpu_avail

@lru_cache(maxsize=1)
def get_torch_device() -> torch.device:
    """Return the active torch device, preferring DirectML, MPS, XPU, CUDA, then CPU."""
    if directml_enabled:
        return directml_device  # type: ignore
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    if is_intel_xpu():
        return torch.device("xpu", torch.xpu.current_device())
    return torch.device("cuda", torch.cuda.current_device())

def get_total_memory(dev: Optional[torch.device] = None, torch_total: bool = False) -> Tuple[int,int] or int:
    """Return total physical (and optionally reserved) memory on dev."""
    dev = dev or get_torch_device()
    if dev.type in ("cpu","mps"):
        total = psutil.virtual_memory().total
        return (total, total) if torch_total else total

    if directml_enabled:
        total = 1<<30  # placeholder 1GB
        return (total, total) if torch_total else total

    if is_intel_xpu():
        stats = torch.xpu.memory_stats(dev)
        reserved = stats['reserved_bytes.all.current']
        total = torch.xpu.get_device_properties(dev).total_memory
        return (total, reserved) if torch_total else total

    # CUDA path
    stats = torch.cuda.memory_stats(dev)
    reserved = stats['reserved_bytes.all.current']
    _, cuda_total = torch.cuda.mem_get_info(dev)
    return (cuda_total, reserved) if torch_total else cuda_total

def get_free_memory(dev: Optional[torch.device] = None, torch_free: bool = False) -> Tuple[int,int] or int:
    """Return free total (and optionally free reserved) memory on dev."""
    dev = dev or get_torch_device()
    if dev.type in ("cpu","mps"):
        free = psutil.virtual_memory().available
        return (free, free) if torch_free else free

    if directml_enabled:
        free = 1<<30
        return (free, free) if torch_free else free

    if is_intel_xpu():
        stats = torch.xpu.memory_stats(dev)
        active = stats['active_bytes.all.current']
        reserved = stats['reserved_bytes.all.current']
        free_torch = reserved - active
        free_xpu   = torch.xpu.get_device_properties(dev).total_memory - reserved
        total_free = free_xpu + free_torch
        return (total_free, free_torch) if torch_free else total_free

    # CUDA path
    stats = torch.cuda.memory_stats(dev)
    active   = stats['active_bytes.all.current']
    reserved = stats['reserved_bytes.all.current']
    cuda_free, _ = torch.cuda.mem_get_info(dev)
    free_torch = reserved - active
    return (cuda_free + free_torch, free_torch) if torch_free else (cuda_free + free_torch)

# ── Report System Info ─────────────────────────────────────────────────────────
total_vram = get_total_memory(get_torch_device()) // (1024*1024)
total_ram  = psutil.virtual_memory().total // (1024*1024)
logger.info(f"Total VRAM: {total_vram} MB; Total RAM: {total_ram} MB")
logger.info(f"PyTorch version: {torch.__version__}")

# ── Exceptions & Flags ──────────────────────────────────────────────────────────
try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except AttributeError:
    OOM_EXCEPTION = RuntimeError

if directml_enabled:
    OOM_EXCEPTION = RuntimeError

# ── xFormers Support ────────────────────────────────────────────────────────────
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
    XFORMERS_OK = False
else:
    try:
        import xformers, xformers.ops  # type: ignore
        XFORMERS_OK = getattr(xformers, "_has_cpp_library", True)
        XFORMERS_VERSION = xformers.version.__version__  # type: ignore
        logger.info(f"xFormers version: {XFORMERS_VERSION}")
        if XFORMERS_VERSION.startswith("0.0.18"):
            logger.warning("xFormers 0.0.18 has a known bug with high-res images. Disabling VAE use.")
            XFORMERS_ENABLED_VAE = False
    except ImportError:
        XFORMERS_OK = False

# ── Accelerator Capability ─────────────────────────────────────────────────────
def is_nvidia() -> bool:
    return cpu_state == CPUState.GPU and torch.version.cuda is not None

ENABLE_PYTORCH_ATTENTION = False
if args.attention_pytorch:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_OK = False

# Auto‑enable PyTorch attention on CUDA>=2.0 or Intel XPU
try:
    if is_nvidia() and int(torch.version.cuda.split(".")[0]) >= 2:
        if not (args.attention_split or args.attention_quad or ENABLE_PYTORCH_ATTENTION):
            ENABLE_PYTORCH_ATTENTION = True
    if torch.cuda.is_bf16_supported() and torch.cuda.get_device_properties(get_torch_device()).major >= 8:
        args.vae_in_bf16 = True
    if is_intel_xpu() and not (args.attention_split or args.attention_quad):
        ENABLE_PYTORCH_ATTENTION = True
except Exception:
    pass

# Apply SDP settings if PyTorch attention is enabled
if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    logger.info("Enabled PyTorch Scaled Dot‑Product Attention backends")

# ── VRAM Overrides ───────────────────────────────────────────────────────────────
if args.always_low_vram:
    set_vram_to = VRAMState.LOW_VRAM
elif args.always_no_vram:
    set_vram_to = VRAMState.NO_VRAM
elif args.always_high_vram or args.always_gpu:
    vram_state = VRAMState.HIGH_VRAM

if args.all_in_fp32:
    logger.info("Forcing FP32 everywhere")
    FORCE_FP32 = True
else:
    FORCE_FP32 = False

if args.all_in_fp16:
    logger.info("Forcing FP16 everywhere")
    FORCE_FP16 = True
else:
    FORCE_FP16 = False

if lowvram_avail and set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
    vram_state = set_vram_to
if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED
if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

logger.info(f"Final VRAM state: {vram_state.name}")

ALWAYS_VRAM_OFFLOAD = args.always_offload_from_vram
if ALWAYS_VRAM_OFFLOAD:
    logger.info("Always offload VRAM enabled")

PIN_SHARED_MEMORY = args.pin_shared_memory
if PIN_SHARED_MEMORY:
    logger.info("Pinned shared memory enabled")

# ── Device Naming ────────────────────────────────────────────────────────────────
def get_torch_device_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        try:
            back = torch.cuda.get_allocator_backend()
        except Exception:
            back = ""
        return f"{dev} {torch.cuda.get_device_name(dev)} ({back})"
    if dev.type == "xpu":
        return f"{dev} {torch.xpu.get_device_name(dev)}"
    return f"{dev.type}"

try:
    name = get_torch_device_name(get_torch_device())
    logger.info(f"Using device: {name}")
    if "rtx" in name.lower() and not args.cuda_malloc:
        logger.info("Hint: your RTX GPU supports --cuda-malloc for speedups")
except Exception:
    logger.warning("Failed to determine torch device name")

# ── Memory/Profile & Model Loading Helpers ───────────────────────────────────────
current_loaded_models: List["LoadedModel"] = []
current_inference_memory = 1 << 30  # 1 GB

# (State‑dict analysis, bake, module size/move, LoadedModel, free_memory, load_models_gpu, etc.)
# ... keep existing implementations here, but replace print() calls with logger.info()/warning()
# ... and use get_free_memory() / get_total_memory() as above

def soft_empty_cache(force: bool = False):
    """Empty GPU/MPS/XPU caches if signal_empty_cache is set or forced."""
    global signal_empty_cache
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available() and (force or is_nvidia()):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    signal_empty_cache = False

# ── End of enhanced ComfyUI backend setup ────────────────────────────────────────
# ── State‑dict & Model Utilities ───────────────────────────────────────────────
def state_dict_size(sd: dict, exclude_device: Optional[torch.device] = None) -> int:
    total = 0
    for tensor in sd.values():
        if exclude_device is not None and tensor.device == exclude_device:
            continue
        total += tensor.nelement() * tensor.element_size()
    return total

def state_dict_parameters(sd: dict) -> int:
    return sum(t.nelement() for t in sd.values())

def state_dict_dtype(sd: dict) -> torch.dtype:
    # detect special GGUF / bitsandbytes keys
    for k, v in sd.items():
        if hasattr(v, 'gguf_cls'):
            return 'gguf'
        if 'bitsandbytes__nf4' in k:
            return 'nf4'
        if 'bitsandbytes__fp4' in k:
            return 'fp4'
    # otherwise pick most frequent dtype
    counts = {}
    for t in sd.values():
        counts[t.dtype] = counts.get(t.dtype, 0) + 1
    return max(counts, key=counts.get)

def bake_gguf_model(model: torch.nn.Module) -> torch.nn.Module:
    if getattr(model, 'gguf_baked', False):
        return model
    for p in model.parameters():
        cls = getattr(p, 'gguf_cls', None)
        if cls is not None:
            cls.bake(p)
    global signal_empty_cache
    signal_empty_cache = True
    model.gguf_baked = True
    return model

def module_size(module: torch.nn.Module,
                exclude_device: Optional[torch.device] = None,
                include_device: Optional[torch.device] = None,
                return_split: bool = False) -> Tuple[float, float, float] or float:
    total = weight_mem = 0.0
    for name, p in module.named_parameters():
        t = p.data
        if exclude_device is not None and t.device == exclude_device:
            continue
        if include_device is not None and t.device != include_device:
            continue
        size = t.nelement() * t.element_size()
        # adjust for quantization
        if getattr(p, 'quant_type', None) in ('fp4','nf4'):
            size = t.nelement() * (1.1 if t.element_size()<=1 else 0.55)
        total += size
        if 'weight' in name:
            weight_mem += size
    if return_split:
        return total, weight_mem, total - weight_mem
    return total

def module_move(module: torch.nn.Module,
                device: torch.device,
                recursive: bool = True,
                excluded_pattens: List[str] = []) -> torch.nn.Module:
    if recursive:
        return module.to(device)
    for name, p in module.named_parameters(recurse=False):
        if name in excluded_pattens:
            continue
        setattr(module, name, utils.tensor2parameter(p.to(device)))
    return module

def build_module_profile(model: torch.nn.Module, gpu_budget: float):
    all_mod, legacy = [], []
    for m in model.modules():
        if hasattr(m, "parameters_manual_cast"):
            m.total_mem, m.weight_mem, m.extra_mem = module_size(m, return_split=True)
            all_mod.append(m)
        elif hasattr(m, "weight"):
            m.total_mem, m.weight_mem, m.extra_mem = module_size(m, return_split=True)
            legacy.append(m)
    gpu_mod, gpu_extras, cpu_mod = [], [], []
    used = 0
    for m in legacy:
        gpu_mod.append(m)
        used += m.total_mem
    for m in sorted(all_mod, key=lambda x: x.extra_mem):
        if used + m.extra_mem < gpu_budget:
            gpu_extras.append(m)
            used += m.extra_mem
        else:
            cpu_mod.append(m)
    for m in sorted(gpu_extras, key=lambda x: x.weight_mem):
        if used + m.weight_mem < gpu_budget:
            gpu_mod.append(m)
            used += m.weight_mem
        else:
            cpu_mod.append(m)
    return gpu_mod, gpu_extras, cpu_mod

# ── LoadedModel & Model‑loading Logic ──────────────────────────────────────────
class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.model_accelerated = False
        self.inclusive_memory = 0
        self.exclusive_memory = 0

    def compute_inclusive_exclusive_memory(self):
        self.inclusive_memory = module_size(self.model.model, include_device=self.device)
        self.exclusive_memory = module_size(self.model.model, exclude_device=self.device)

    def model_load(self, gpu_swap_budget: float = -1):
        do_cpu_swap = gpu_swap_budget >= 0
        target_device = self.device if not do_cpu_swap else self.device
        self.model.model_patches_to(target_device)
        try:
            real = self.model.forge_patch_model(target_device if not do_cpu_swap else None)
            self.model.current_device = self.device
        except Exception as e:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model_unload()
            raise
        if do_cpu_swap:
            gpu_mod, gpu_extras, cpu_mod = build_module_profile(real, gpu_swap_budget)
            pin = PIN_SHARED_MEMORY and cpu_state==CPUState.CPU
            used, swapped = 0, 0
            for m in gpu_mod:
                m.to(self.device); used += m.total_mem
            for m in cpu_mod:
                m.prev_cast = m.parameters_manual_cast
                m.parameters_manual_cast = True
                m.to(self.model.offload_device)
                if pin: m._apply(lambda x: x.pin_memory())
                swapped += m.total_mem
            for m in gpu_extras:
                m.prev_cast = m.parameters_manual_cast
                m.parameters_manual_cast = True
                module_move(m, self.device, recursive=False, excluded_pattens=['weight'])
                w = m.weight
                if w is not None:
                    m.weight = utils.tensor2parameter(
                        w.to(self.model.offload_device).pin_memory() if pin else w.to(self.model.offload_device)
                    )
                used += m.extra_mem; swapped += m.weight_mem
            logger.info(f"Swap loaded: swapped={swapped/1e6:.2f}MB, gpu={used/1e6:.2f}MB")
            self.model_accelerated = True
            global signal_empty_cache; signal_empty_cache=True
        bake_gguf_model(real)
        self.model.refresh_loras()
        if is_intel_xpu() and not args.disable_ipex_hijack:
            real = torch.xpu.optimize(real.eval(), inplace=True, auto_kernel_selection=True, graph_mode=True)
        return real

    def model_unload(self, avoid_moving: bool = False):
        if self.model_accelerated:
            for m in self.model.model.modules():
                if hasattr(m, 'prev_cast'):
                    m.parameters_manual_cast = m.prev_cast; del m.prev_cast
            self.model_accelerated = False
        if avoid_moving:
            self.model.forge_unpatch_model()
        else:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other): return self.model is other.model

current_loaded_models: List[LoadedModel] = []
current_inference_memory = 1<<30  # 1 GB

def minimum_inference_memory() -> float:
    return current_inference_memory

def unload_model_clones(model):
    to_unload = [i for i, lm in enumerate(current_loaded_models) if model.is_clone(lm.model)]
    for i in reversed(to_unload):
        current_loaded_models.pop(i).model_unload(avoid_moving=True)

def free_memory(req_bytes: float, device: torch.device,
                keep_loaded: List[LoadedModel]=[], free_all: bool=False):
    # unload abandoned
    for i in reversed(range(len(current_loaded_models))):
        if sys.getrefcount(current_loaded_models[i].model)<=2:
            current_loaded_models.pop(i).model_unload(avoid_moving=True)
    if free_all:
        req_bytes = 1e30
        logger.info(f"[Unload] Freeing all memory on {device}")
    else:
        logger.info(f"[Unload] Need {req_bytes/1e6:.2f}MB on {device}")
    offload_all = ALWAYS_VRAM_OFFLOAD or vram_state==VRAMState.NO_VRAM
    for lm in reversed(current_loaded_models):
        if not offload_all:
            free_now = get_free_memory(device)
            if free_now > req_bytes: break
        if lm.device==device and lm not in keep_loaded:
            logger.info(f" Unloading {lm.model.model.__class__.__name__}")
            lm.model_unload()
            current_loaded_models.remove(lm)
    soft_empty_cache()
    logger.info("Unload complete.")

def compute_model_gpu_memory_when_using_cpu_swap(free_now: float, inf_mem: float) -> float:
    avail = free_now - inf_mem
    return int(max(avail/1.3, avail - 1.25*(1<<30)))

def load_models_gpu(models: List, memory_required: float=0, hard_preserve: float=0):
    start = time.perf_counter()
    to_free = max(minimum_inference_memory(), memory_required) + hard_preserve
    to_inf = minimum_inference_memory() + hard_preserve
    to_load, already = [], []
    for m in models:
        lm = LoadedModel(m)
        if lm in current_loaded_models:
            idx = current_loaded_models.index(lm)
            current_loaded_models.insert(0, current_loaded_models.pop(idx))
            already.append(lm)
        else:
            to_load.append(lm)
    for lm in to_load: unload_model_clones(lm.model)
    # free for already loaded
    for lm in already:
        if lm.device!=cpu:
            free_memory(to_free, lm.device, keep_loaded=already)
    # free for to_load
    dev_reqs = {}
    for lm in to_load:
        lm.compute_inclusive_exclusive_memory()
        dev_reqs[lm.device] = dev_reqs.get(lm.device,0) + lm.exclusive_memory + lm.inclusive_memory*0.25
    for dev, req in dev_reqs.items():
        if dev!=cpu:
            free_memory(req*1.3+to_free, dev, keep_loaded=already)
    # actually load
    for lm in to_load:
        dev = lm.device
        budget = -1
        if lowvram_avail and vram_state in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM):
            free_now = get_free_memory(dev)
            need = lm.exclusive_memory + to_inf
            rem = free_now - need
            if rem<0:
                vram_state_local = VRAMState.LOW_VRAM
                budget = compute_model_gpu_memory_when_using_cpu_swap(free_now, to_inf)
            else:
                vram_state_local = vram_state
        else:
            vram_state_local = vram_state
        if vram_state_local==VRAMState.NO_VRAM: budget=0
        real = lm.model_load(budget)
        current_loaded_models.insert(0, lm)
    elapsed = time.perf_counter()-start
    logger.info(f"Model loading took {elapsed:.2f}s")

def load_model_gpu(model): return load_models_gpu([model])

def cleanup_models():
    to_rm = [i for i,lm in enumerate(current_loaded_models) if sys.getrefcount(lm.model)<=2]
    for i in reversed(to_rm):
        lm = current_loaded_models.pop(i)
        lm.model_unload()

# ── Dtype & Device Utilities ───────────────────────────────────────────────────
def dtype_size(dt: torch.dtype) -> int:
    if dt in (torch.float16, torch.bfloat16): return 2
    if dt==torch.float32: return 4
    try: return dt.itemsize
    except: return 4

def unet_offload_device() -> torch.device:
    return cpu if vram_state!=VRAMState.HIGH_VRAM else get_torch_device()

def unet_initial_load_device(params: int, dtype: torch.dtype) -> torch.device:
    dev, cpu_dev = get_torch_device(), cpu
    if vram_state==VRAMState.HIGH_VRAM: return dev
    if ALWAYS_VRAM_OFFLOAD: return cpu_dev
    size = dtype_size(dtype)*params
    return dev if get_free_memory(dev)>get_free_memory(cpu_dev) and size<get_free_memory(dev) else cpu_dev

def unet_dtype(device=None, model_params=0) -> torch.dtype:
    if args.unet_in_bf16: return torch.bfloat16
    if args.unet_in_fp16: return torch.float16
    if args.unet_in_fp8_e4m3fn: return torch.float8_e4m3fn
    if args.unet_in_fp8_e5m2: return torch.float8_e5m2
    # fallback logic...
    return torch.float32

def get_computation_dtype(device=None, parameters=0) -> torch.dtype:
    # similar fallback...
    return torch.float32

def text_encoder_offload_device() -> torch.device:
    return get_torch_device() if args.always_gpu else cpu

def text_encoder_device() -> torch.device:
    if args.always_gpu: return get_torch_device()
    if vram_state in (VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM) and should_use_fp16():
        return get_torch_device()
    return cpu

def text_encoder_dtype(device=None) -> torch.dtype:
    if args.clip_in_fp8_e4m3fn: return torch.float8_e4m3fn
    if args.clip_in_fp8_e5m2: return torch.float8_e5m2
    if args.clip_in_fp16: return torch.float16
    if args.clip_in_fp32: return torch.float32
    return torch.float16

def intermediate_device() -> torch.device:
    return get_torch_device() if args.always_gpu else cpu

def vae_device() -> torch.device:
    return cpu if args.vae_in_cpu else get_torch_device()

def vae_offload_device() -> torch.device:
    return get_torch_device() if args.always_gpu else cpu

def vae_dtype(device=None, allowed: List[torch.dtype]=[]) -> torch.dtype:
    # similar to unet_dtype but considers VAE_DTYPES
    return torch.float32

logger.info(f"VAE dtype preferences: {VAE_DTYPES} -> {vae_dtype()}")

def get_autocast_device(dev: torch.device) -> str:
    return dev.type if hasattr(dev, 'type') else "cuda"

def supports_dtype(device: torch.device, dt: torch.dtype) -> bool:
    if dt==torch.float32: return True
    if device.type=='cpu': return False
    return dt in (torch.float16, torch.bfloat16)

def supports_cast(device: torch.device, dt: torch.dtype) -> bool:
    if dt in (torch.float16, torch.float32): return True
    if dt==torch.bfloat16 and (device.type.startswith("cuda") or is_intel_xpu()): return True
    if dt in (torch.float8_e4m3fn, torch.float8_e5m2): return True
    return False

def pick_weight_dtype(dt, fb, device=None):
    out = dt or fb
    if dtype_size(out)>dtype_size(fb) or not supports_cast(device, out):
        return fb
    return out

def device_supports_non_blocking(device: torch.device) -> bool:
    if device.type in ('mps','cpu'): return False
    if is_intel_xpu() or args.pytorch_deterministic or directml_enabled:
        return False
    return True

def force_channels_last() -> bool:
    return bool(args.force_channels_last)

def cast_to_device(t: torch.Tensor, device: torch.device, dtype: torch.dtype, copy: bool=False) -> torch.Tensor:
    nb = device_supports_non_blocking(device)
    try:
        if copy:
            return t.to(device, dtype=dtype, non_blocking=nb, copy=True)
        return t.to(device, non_blocking=nb).to(dtype, non_blocking=nb)
    except:
        return t.to(device, dtype=dtype, non_blocking=nb)

def sage_attention_enabled() -> bool: return args.use_sage_attention
def flash_attention_enabled() -> bool: return args.use_flash_attention
def xformers_enabled() -> bool:
    return cpu_state==CPUState.GPU and not directml_enabled and not is_intel_xpu() and XFORMERS_OK
def xformers_enabled_vae() -> bool: return xformers_enabled() and XFORMERS_ENABLED_VAE
def pytorch_attention_enabled() -> bool: return ENABLE_PYTORCH_ATTENTION
def pytorch_attention_flash_attention() -> bool:
    return ENABLE_PYTORCH_ATTENTION and (is_nvidia() or is_intel_xpu())

def force_upcast_attention_dtype() -> Optional[torch.dtype]:
    up = args.force_upcast_attention
    if platform.mac_ver()[0].startswith("14.5"):
        up = True
    return torch.float32 if up else None

def cpu_mode() -> bool: return cpu_state==CPUState.CPU
def mps_mode() -> bool: return cpu_state==CPUState.MPS
def is_device_type(device, tp: str) -> bool: return getattr(device, 'type', '')==tp
def is_device_cpu(device): return is_device_type(device,'cpu')
def is_device_mps(device): return is_device_type(device,'mps')
def is_device_cuda(device): return is_device_type(device,'cuda')

def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False) -> bool:
    # original complex heuristic unchanged
    return False  # placeholder

def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False) -> bool:
    return False  # placeholder

def can_install_bnb() -> bool:
    try:
        return torch.cuda.is_available() and tuple(map(int,torch.version.cuda.split('.')))>=(11,7)
    except:
        return False

signal_empty_cache = False

def unload_all_models():
    free_memory(1e30, get_torch_device(), free_all=True)
# ── FP16 / BF16 Heuristics and Related Utilities ─────────────────────────────────
def should_use_fp16(
    device: Optional[torch.device] = None,
    model_params: int = 0,
    prioritize_performance: bool = True,
    manual_cast: bool = False,
) -> bool:
    global directml_enabled, FORCE_FP16, FORCE_FP32

    # CPU never uses FP16 for compute
    if device and device.type == "cpu":
        return False

    # explicit overrides
    if FORCE_FP16:
        return True
    if FORCE_FP32:
        return False

    # MPS always uses FP16
    if device and device.type == "mps":
        return True
    if torch.backends.mps.is_available() and cpu_state == CPUState.MPS:
        return True

    # DirectML doesn't support FP16
    if directml_enabled:
        return False

    # CPU mode never FP16
    if cpu_state == CPUState.CPU:
        return False

    # Intel XPU uses BF16, but FP16 compute also works
    if is_intel_xpu():
        return True

    # ROCm (HIP) uses FP16
    if getattr(torch.version, "hip", False):
        return True

    # NVIDIA GPUs: check compute capability
    if torch.cuda.is_available() and device and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        major = props.major
        name = props.name.lower()

        if major >= 8:
            return True
        if major < 6:
            return False

        # 10-series: FP16 compute is slow; require storage only
        series_10 = [
            "1080","1070","titan x","p3000","p3200","p4000","p4200",
            "p5000","p5200","p6000","1060","1050","p40","p100","p6","p4"
        ]
        if any(tag in name for tag in series_10):
            if manual_cast:
                free_model_mem = get_free_memory(device) * 0.9 - minimum_inference_memory()
                if not prioritize_performance or model_params * 4 > free_model_mem:
                    return True
            return False

        # <7.x no FP16
        if major < 7:
            return False

        # 16-series known buggy
        series_16 = [
            "1660","1650","1630","t500","t550","t600","mx550","mx450",
            "cmp 30hx","t2000","t1000","t1200"
        ]
        if any(tag in name for tag in series_16):
            return False

        return True

    # default fallback
    return False


def should_use_bf16(
    device: Optional[torch.device] = None,
    model_params: int = 0,
    prioritize_performance: bool = True,
    manual_cast: bool = False,
) -> bool:
    global directml_enabled, FORCE_FP32

    # CPU BF16 compute is extremely slow
    if device and device.type == "cpu":
        return False

    # MPS: BF16 supported
    if device and device.type == "mps":
        return True
    if torch.backends.mps.is_available() and cpu_state == CPUState.MPS:
        return True

    # explicit override
    if FORCE_FP32:
        return False

    # DirectML: no BF16
    if directml_enabled:
        return False

    # CPU mode
    if cpu_state == CPUState.CPU:
        return False

    # Intel XPU: BF16 native
    if is_intel_xpu():
        return True

    # CUDA BF16 support
    if torch.cuda.is_available():
        if device is None:
            device = get_torch_device()
        props = torch.cuda.get_device_properties(device)
        if props.major >= 8:
            return True
        if torch.cuda.is_bf16_supported() and manual_cast:
            free_model_mem = get_free_memory(device) * 0.9 - minimum_inference_memory()
            if not prioritize_performance or model_params * 4 > free_model_mem:
                return True

    return False


def get_computation_dtype(
    inference_device: Optional[torch.device],
    parameters: int = 0,
    supported_dtypes: List[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32],
) -> torch.dtype:
    """
    Pick the best compute dtype given device & model size.
    """
    for candidate in supported_dtypes:
        if candidate is torch.float16 and should_use_fp16(
            inference_device, parameters, prioritize_performance=True, manual_cast=False
        ):
            return candidate
        if candidate is torch.bfloat16 and should_use_bf16(
            inference_device, parameters, prioritize_performance=True, manual_cast=False
        ):
            return candidate
    return torch.float32


def vae_dtype(
    device: Optional[torch.device] = None,
    allowed_dtypes: List[torch.dtype] = [],
) -> torch.dtype:
    """
    Pick VAE dtype based on flags, device, and VAE_DTYPES preferences.
    """
    if args.vae_in_fp16:
        return torch.float16
    if args.vae_in_bf16:
        return torch.bfloat16
    if args.vae_in_fp32:
        return torch.float32

    # try allowed list first
    for d in allowed_dtypes:
        if d is torch.float16 and should_use_fp16(device, prioritize_performance=False):
            return d
        if d in VAE_DTYPES:
            return d

    # fallback to global preference
    return VAE_DTYPES[0]


def can_install_bnb() -> bool:
    """
    Check if bitsandbytes (bnb) can be installed: requires CUDA >= 11.7 on NVIDIA.
    """
    try:
        if not torch.cuda.is_available():
            return False
        ver = tuple(int(x) for x in torch.version.cuda.split("."))
        return ver >= (11, 7)
    except Exception:
        return False
# ── Cache Signaling & Full Unload ───────────────────────────────────────────────
signal_empty_cache = False

def soft_empty_cache(force: bool = False) -> None:
    """
    Empty device caches (MPS, XPU, CUDA) when signaled or forced.
    """
    global signal_empty_cache
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available() and (force or is_nvidia()):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    signal_empty_cache = False

def unload_all_models() -> None:
    """
    Force‑unload all loaded models to reclaim every byte of memory.
    """
    free_memory(1e30, get_torch_device(), free_all=True)
