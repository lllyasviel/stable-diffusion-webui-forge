import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Callable
from functools import lru_cache

import gradio as gr
from modules import shared_cmd_options, shared_gradio_themes, options, shared_items, sd_models_types
from modules.paths_internal import (
    models_path, script_path, data_path,
    sd_configs_path, sd_default_config,
    extensions_dir, extensions_builtin_dir
)
from modules import util
from backend import memory_management

# ── Logging ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ── Configuration Dataclass ────────────────────────────────────────────────────
@dataclass
class UIConfig:
    styles_files: List[Path] = field(default_factory=lambda: [
        Path(data_path) / "styles.csv",
        Path(data_path) / "styles_integrated.csv",
    ])
    ui_settings_file: Path = Path(shared_cmd_options.cmd_opts.ui_settings_file)
    hide_ui_dirs: bool = shared_cmd_options.cmd_opts.hide_ui_dir_config
    hf_endpoint: str = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    xformers_available: bool = memory_management.xformers_enabled()

# Instantiate UI configuration
cmd_opts = shared_cmd_options.cmd_opts
parser   = shared_cmd_options.parser
config   = UIConfig()

# ── Globals & Placeholders ─────────────────────────────────────────────────────
demo:        Optional[gr.Blocks]         = None
device:      Optional[str]               = None
weight_loc:  Optional[str]               = None
hypernetworks:       Dict[str, object]   = {}
loaded_hypernetworks: List[str]          = []
state:       Optional['shared_state.State']          = None
prompt_styles: Optional['styles.StyleDatabase']      = None
interrogator: Optional['interrogate.InterrogateModels'] = None
face_restorers:     List[object]         = []
options_templates:  Dict[str, options.OptionInfo]   = {}
opts:        options.Options               = options.opts
restricted_opts: Set[str]                  = set(options.restricted_opts or [])
sd_model:    Optional[object]              = None
settings_components: Dict[str, gr.Component] = {}
tab_names:   List[str]                    = []

latent_upscale_default_mode = "Latent"
latent_upscale_modes: Dict[str, Dict[str, bool]] = {
    "Latent":                {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)":  {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)":      {"mode": "bicubic",  "antialias": False},
    "Latent (bicubic aa)":   {"mode": "bicubic",  "antialias": True},
    "Latent (nearest)":      {"mode": "nearest",  "antialias": False},
    "Latent (nearest-exact)":{"mode": "nearest-exact","antialias": False},
}

sd_upscalers: List[str]      = []
clip_model:  Optional[object] = None
progress_print_out = sys.stdout
gradio_theme       = shared_gradio_themes.reload_gradio_theme(config.ui_settings_file)
total_tqdm:        'shared_total_tqdm.TotalTQDM'    = None
mem_mon:           'memmon.MemUsageMonitor'        = None

# aliases for convenience
list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints   = shared_items.refresh_checkpoints
list_samplers         = shared_items.list_samplers
reload_hypernetworks  = shared_items.reload_hypernetworks

hf_endpoint = config.hf_endpoint

# ── Caching & Utilities ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_checkpoint_tiles() -> List[str]:
    return list_checkpoint_tiles()

@lru_cache(maxsize=None)
def list_models(model_type: sd_models_types.ModelType) -> List[str]:
    try:
        return util.listfiles(models_path, model_type.file_ext)
    except Exception as e:
        logger.error(f"Error listing {model_type}: {e}")
        return []

# ── Directory Watcher for Extensions ────────────────────────────────────────────
def watch_extensions_directory(on_change: Callable) -> Optional[object]:
    try:
        from watchdog.observers import Observer
        from watchdog.events    import FileSystemEventHandler
    except ImportError:
        logger.warning("watchdog not installed; extension auto‑reload disabled")
        return None

    class ExtHandler(FileSystemEventHandler):
        def on_any_event(self, event):
            logger.info(f"Extensions directory changed: {event.src_path}")
            on_change()

    observer = Observer()
    observer.schedule(ExtHandler(), str(extensions_dir), recursive=True)
    observer.start()
    return observer

_ext_observer = watch_extensions_directory(reload_hypernetworks)

# ── Hide/Show UI Directories ────────────────────────────────────────────────────
hide_dirs = {"visible": not config.hide_ui_dirs}

# ── Initialization Logging ─────────────────────────────────────────────────────
logger.info(f"Styles files: {config.styles_files}")
logger.info(f"Hiding UI dirs: {config.hide_ui_dirs}")
logger.info(f"HuggingFace endpoint: {config.hf_endpoint}")
logger.info(f"xFormers available: {config.xformers_available}")
