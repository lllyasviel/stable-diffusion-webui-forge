import os
import sys
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, cwd  # noqa: F401


sys.path.insert(0, script_path)

sd_path = os.path.dirname(__file__)

path_dirs = [
    (os.path.join(sd_path, '../repositories/BLIP'), 'models/blip.py', 'BLIP', []),
    (os.path.join(sd_path, '../packages_3rdparty'), 'gguf/quants.py', 'packages_3rdparty', []),
    # (os.path.join(sd_path, '../repositories/k-diffusion'), 'k_diffusion/sampling.py', 'k_diffusion', ["atstart"]),
    (os.path.join(sd_path, '../repositories/huggingface_guess'), 'huggingface_guess/detection.py', 'huggingface_guess', []),
]

paths = {}

for d, must_exist, what, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        d = os.path.abspath(d)
        if "atstart" in options:
            sys.path.insert(0, d)
        else:
            sys.path.append(d)
        paths[what] = d
