import torch
import os
import time
import httpx
import warnings
import gradio.networking
import safetensors.torch

from pathlib import Path
from tqdm import tqdm
from modules import timer

# Add a module-level timer for this patch file
patch_basic_timer = timer.Timer(print_log=True)

# Example: record import time
patch_basic_timer.record("module_import")


def gradio_url_ok_fix(url: str) -> bool:
    with patch_basic_timer.subcategory("gradio_url_ok_fix"):
        try:
            for _ in range(5):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    r = httpx.head(url, timeout=999, verify=False)
                if r.status_code in (200, 401, 302):
                    return True
                time.sleep(0.500)
        except (ConnectionError, httpx.ConnectError):
            return False
        return False


def build_loaded(module, loader_name):
    original_loader_name = loader_name + '_origin'

    if not hasattr(module, original_loader_name):
        setattr(module, original_loader_name, getattr(module, loader_name))

    original_loader = getattr(module, original_loader_name)

    def loader(*args, **kwargs):
        with patch_basic_timer.subcategory(f"{module.__name__}.{loader_name}"):
            result = None
            try:
                result = original_loader(*args, **kwargs)
            except Exception as e:
                result = None
                exp = str(e) + '\n'
                for path in list(args) + list(kwargs.values()):
                    if isinstance(path, str):
                        if os.path.exists(path):
                            exp += f'File corrupted: {path} \n'
                            corrupted_backup_file = path + '.corrupted'
                            if os.path.exists(corrupted_backup_file):
                                os.remove(corrupted_backup_file)
                            os.replace(path, corrupted_backup_file)
                            if os.path.exists(path):
                                os.remove(path)
                            exp += f'Forge has tried to move the corrupted file to {corrupted_backup_file} \n'
                            exp += f'You may try again now and Forge will download models again. \n'
                raise ValueError(exp)
            return result

    setattr(module, loader_name, loader)
    return


def always_show_tqdm(*args, **kwargs):
    kwargs['disable'] = False
    if 'name' in kwargs:
        del kwargs['name']
    return tqdm(*args, **kwargs)


def long_path_prefix(path: Path) -> Path:
    if os.name == 'nt' and not str(path).startswith("\\\\?\\") and not path.exists():
        return Path("\\\\?\\" + str(path))
    return path


def patch_all_basics():
    # Import timer for profiling
    from modules import timer
    startup_timer = timer.startup_timer
    with startup_timer.subcategory("patch_basic.py entry"):
        patch_basic_timer.record("patch_all_basics entry")
        with startup_timer.subcategory("huggingface imports"):
            import logging
            from huggingface_hub import file_download
            file_download.tqdm = always_show_tqdm
            startup_timer.record("huggingface_hub import")
            from transformers.dynamic_module_utils import logger
            logger.setLevel(logging.ERROR)
            startup_timer.record("transformers import")
        with startup_timer.subcategory("download patches"):
            from huggingface_hub.file_download import _download_to_tmp_and_move as original_download_to_tmp_and_move
            def patched_download_to_tmp_and_move(incomplete_path, destination_path, url_to_download, proxies, headers, expected_size, filename, force_download):
                incomplete_path = long_path_prefix(incomplete_path)
                destination_path = long_path_prefix(destination_path)
                return original_download_to_tmp_and_move(incomplete_path, destination_path, url_to_download, proxies, headers, expected_size, filename, force_download)
            file_download._download_to_tmp_and_move = patched_download_to_tmp_and_move
            startup_timer.record("file download patches")
        with startup_timer.subcategory("loader patches"):
            gradio.networking.url_ok = gradio_url_ok_fix
            startup_timer.record("gradio networking patch")
            build_loaded(safetensors.torch, 'load_file')
            startup_timer.record("safetensors patch")
            build_loaded(torch, 'load')
            startup_timer.record("torch load patch")
        patch_basic_timer.record("patch_all_basics exit")
    return
