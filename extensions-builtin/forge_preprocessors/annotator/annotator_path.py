import os
from modules_forge.shared import preprocessor_dir


models_path = preprocessor_dir

clip_vision_path = os.path.join(preprocessor_dir, 'clip_vision')
os.makedirs(preprocessor_dir, exist_ok=True)

models_path = os.path.realpath(models_path)
os.makedirs(models_path, exist_ok=True)
