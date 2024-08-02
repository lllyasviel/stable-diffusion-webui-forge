import torch
import os
import time
import httpx
import warnings
import gradio.networking
import safetensors.torch


def gradio_url_ok_fix(url: str) -> bool:
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


def patch_all_basics():
    gradio.networking.url_ok = gradio_url_ok_fix
    build_loaded(safetensors.torch, 'load_file')
    build_loaded(torch, 'load')
    return
