import os
from modules import shared
from modules_forge.shared import preprocessor_dir


models_path = preprocessor_dir
if not models_path:
    models_path = getattr(shared.cmd_opts, 'controlnet_annotator_models_path', None)
if not models_path:
    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')

if not os.path.isabs(models_path):
    models_path = os.path.join(shared.data_path, models_path)

clip_vision_path = os.path.join(preprocessor_dir, 'clip_vision')
models_path = os.path.realpath(models_path)

os.makedirs(models_path, exist_ok=True)
os.makedirs(clip_vision_path, exist_ok=True)

print(f'ControlNet preprocessor location: {models_path}')
