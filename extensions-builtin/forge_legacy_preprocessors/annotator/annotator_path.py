import os
from modules_forge.shared import preprocessor_dir


models_path = preprocessor_dir
clip_vision_path = os.path.join(preprocessor_dir, 'clip_vision')

os.makedirs(models_path, exist_ok=True)
os.makedirs(clip_vision_path, exist_ok=True)

print(f'ControlNet preprocessor location: {models_path}')
