import os
import sys
import platform
import subprocess
import importlib.util
from functools import lru_cache
from typing import Iterable, Set, Optional

try:
    # Python 3.8+
    from importlib.metadata import version as _pkg_version  # type: ignore
except Exception:  # pragma: no cover
    _pkg_version = None  # Fallback handled below


# --------------------------------------------------------------------------------------
# GPU detection
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_gpu_names() -> Set[str]:
    """
    Try to enumerate human-readable GPU names across platforms.

    Returns:
        A set of GPU display names (may be empty if detection fails).
    """
    system = os.name
    names: Set[str] = set()

    if system == "nt":
        # Windows: use EnumDisplayDevicesW for Unicode-safe device strings
        import ctypes

        class DISPLAY_DEVICEW(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint),
                ("DeviceName", ctypes.c_wchar * 32),
                ("DeviceString", ctypes.c_wchar * 128),
                ("StateFlags", ctypes.c_uint),
                ("DeviceID", ctypes.c_wchar * 128),
                ("DeviceKey", ctypes.c_wchar * 128),
            ]

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]

        i = 0
        while True:
            dd = DISPLAY_DEVICEW()
            dd.cb = ctypes.sizeof(dd)
            if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):  # type: ignore[attr-defined]
                break
            i += 1
            dev = (dd.DeviceString or "").strip()
            if dev:
                names.add(dev)
        return names

    # Linux/macOS: try nvidia-smi first (fastest/most accurate for NVIDIA)
    def _query_nvidia_smi() -> Iterable[str]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.5,
            )
            for line in out.splitlines():
                line = line.strip()
                if line:
                    yield line
        except Exception:
            return []

    # Linux fallback: lspci
    def _query_lspci() -> Iterable[str]:
        try:
            out = subprocess.check_output(
                ["lspci"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.5,
            )
            for line in out.splitlines():
                if any(k in line.lower() for k in ("vga", "3d", "display")):
                    # Example: 01:00.0 VGA compatible controller: NVIDIA Corporation GA104 ...
                    parts = line.split(":", maxsplit=2)
                    pretty = parts[-1].strip() if parts else line.strip()
                    if pretty:
                        yield pretty
        except Exception:
            return []

    # macOS: system_profiler (slower but useful)
    def _query_system_profiler() -> Iterable[str]:
        if platform.system() != "Darwin":
            return []
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.5,
            )
            for line in out.splitlines():
                line = line.strip()
                # Lines like: "Chipset Model: NVIDIA GeForce GT 750M"
                if line.lower().startswith("chipset model:"):
                    namestr = line.split(":", 1)[-1].strip()
                    if namestr:
                        yield namestr
        except Exception:
            return []

    for source in (_query_nvidia_smi, _query_lspci, _query_system_profiler):
        for n in source():
            names.add(n)

    return names


# --------------------------------------------------------------------------------------
# Policy / blacklist
# --------------------------------------------------------------------------------------
BLACKLIST: Set[str] = {
    "GeForce GTX TITAN X",
    "GeForce GTX 980",
    "GeForce GTX 970",
    "GeForce GTX 960",
    "GeForce GTX 950",
    "GeForce 945M",
    "GeForce 940M",
    "GeForce 930M",
    "GeForce 920M",
    "GeForce 910M",
    "GeForce GTX 750",
    "GeForce GTX 745",
    "Quadro K620",
    "Quadro K1200",
    "Quadro K2200",
    "Quadro M500",
    "Quadro M520",
    "Quadro M600",
    "Quadro M620",
    "Quadro M1000",
    "Quadro M1200",
    "Quadro M2000",
    "Quadro M2200",
    "Quadro M3000",
    "Quadro M4000",
    "Quadro M5000",
    "Quadro M5500",
    "Quadro M6000",
    "GeForce MX110",
    "GeForce MX130",
    "GeForce 830M",
    "GeForce 840M",
    "GeForce GTX 850M",
    "GeForce GTX 860M",
    "GeForce GTX 1650",
    "GeForce GTX 1630",
}

# Allow users to augment/override blacklist at runtime
_env_add = os.environ.get("CUDA_MALLOC_BLACKLIST_ADD", "")
if _env_add.strip():
    for token in _env_add.split(","):
        t = token.strip()
        if t:
            BLACKLIST.add(t)


def _normalize(s: str) -> str:
    return s.lower().strip()


@lru_cache(maxsize=1)
def _has_nvidia_gpu(names: Optional[Iterable[str]] = None) -> bool:
    names = names if names is not None else get_gpu_names()
    for n in names:
        if "nvidia" in _normalize(n):
            return True
    # If we couldn't enumerate, try probing nvidia-smi presence as a heuristic
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL, timeout=1.0)
        return True
    except Exception:
        return False


def _blacklisted(names: Iterable[str]) -> bool:
    lower_names = [_normalize(n) for n in names]
    for b in BLACKLIST:
        b_norm = _normalize(b)
        if any(b_norm in n for n in lower_names):
            return True
    return False


# --------------------------------------------------------------------------------------
# Main decisions
# --------------------------------------------------------------------------------------
def cuda_malloc_supported() -> bool:
    """
    Decide whether to enable PyTorch's cudaMallocAsync backend.

    Rules:
      * Torch must be installed and >= 2.0
      * If an NVIDIA GPU is present and not blacklisted -> supported
      * Env overrides:
          - FORCE_CUDA_MALLOC=1 forces True (even if unknown platform)
          - DISABLE_CUDA_MALLOC=1 forces False
    """
    # Hard overrides
    if os.environ.get("DISABLE_CUDA_MALLOC") == "1":
        return False
    if os.environ.get("FORCE_CUDA_MALLOC") == "1":
        return True

    # Torch >= 2?
    torch_ver = _get_torch_version_str()
    if not torch_ver:
        return False
    if _major_version(torch_ver) < 2:
        return False

    # GPU policy
    try:
        names = get_gpu_names()
    except Exception:
        names = set()

    if not _has_nvidia_gpu(names):
        # No NVIDIA detected -> don't enable by default
        return False

    if _blacklisted(names):
        return False

    return True


def try_cuda_malloc() -> None:
    """
    Attempt to enable cudaMallocAsync for PyTorch by setting
    PYTORCH_CUDA_ALLOC_CONF before torch is imported.

    Prints a short status line and returns None (for backward compatibility).
    """
    # If torch is already imported, warn: changing allocator might have no effect
    if "torch" in sys.modules:
        print("[cudaMallocAsync] Warning: torch is already imported; allocator change may not take effect.")

    do_cuda_malloc = False
    try:
        do_cuda_malloc = cuda_malloc_supported()
    except Exception:
        do_cuda_malloc = False

    if do_cuda_malloc:
        _enable_cuda_malloc_env()
        print("[cudaMallocAsync] Using cudaMallocAsync backend.")
    else:
        print("[cudaMallocAsync] Not enabling cudaMallocAsync backend.")

    return  # keep original None return


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _get_torch_version_str() -> Optional[str]:
    """
    Get torch version string without importing torch (preferred).
    Fallback: read version.py like the original code.
    """
    # Preferred: importlib.metadata
    if _pkg_version:
        try:
            return _pkg_version("torch")
        except Exception:
            pass

    # Fallback: locate torch and read version.py
    try:
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.submodule_search_locations:
            for folder in torch_spec.submodule_search_locations:
                ver_file = os.path.join(folder, "version.py")
                if os.path.isfile(ver_file):
                    spec = importlib.util.spec_from_file_location("torch_version_import", ver_file)
                    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                    assert spec and spec.loader
                    spec.loader.exec_module(module)  # type: ignore[attr-defined]
                    return getattr(module, "__version__", None)
    except Exception:
        return None
    return None


def _major_version(ver: str) -> int:
    # Lightweight robust parsing (avoid extra dependency on packaging)
    try:
        # e.g., "2.3.1+cu121" -> "2"
        return int(ver.split(".", 1)[0].split("+", 1)[0])
    except Exception:
        return 0


def _enable_cuda_malloc_env() -> None:
    """
    Merge/append backend:cudaMallocAsync into PYTORCH_CUDA_ALLOC_CONF without duplicates.
    """
    key = "PYTORCH_CUDA_ALLOC_CONF"
    val = os.environ.get(key, "").strip()

    # Normalize into a set of key:value pairs (comma separated)
    parts = [p.strip() for p in val.split(",") if p.strip()] if val else []
    # If another backend is already specified, replace it; otherwise add ours.
    parts = [p for p in parts if not p.lower().startswith("backend:")]
    parts.append("backend:cudaMallocAsync")

    os.environ[key] = ",".join(parts)
