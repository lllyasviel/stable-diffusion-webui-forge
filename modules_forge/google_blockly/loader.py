import os
import importlib


def load_all_google_blockly():
    current_dir = os.path.dirname(__file__)
    package = __package__

    for filename in os.listdir(current_dir):
        if filename == os.path.basename(__file__):
            continue

        if not filename.endswith(".py"):
            continue

        if filename.endswith("_s.py"):
            continue

        if filename.endswith("_u.py"):
            continue

        if filename.endswith("_m.py"):
            continue

        module_name = f"{package}.{filename[:-3]}"
        importlib.import_module(module_name)

    return
