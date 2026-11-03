import os
import shutil

from diffusers.loaders.single_file_utils import DIFFUSERS_DEFAULT_PIPELINE_PATHS
from huggingface_hub import snapshot_download


token = None
DIFFUSERS_DEFAULT_PIPELINE_PATHS['Kolor'] = {"pretrained_model_name_or_path": "Kwai-Kolors/Kolors"}
DIFFUSERS_DEFAULT_PIPELINE_PATHS['hunyuan'] = {"pretrained_model_name_or_path": "Tencent-Hunyuan/HunyuanDiT-Diffusers"}
DIFFUSERS_DEFAULT_PIPELINE_PATHS['flux'] = {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev"}
DIFFUSERS_DEFAULT_PIPELINE_PATHS['flux_schnell'] = {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-schnell"}

for config_name, config in DIFFUSERS_DEFAULT_PIPELINE_PATHS.items():
    try:
        pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
        local_dir = os.path.join('backend', 'huggingface', pretrained_model_name_or_path)
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(pretrained_model_name_or_path, local_dir=local_dir, allow_patterns=['*.json', '*.txt'],
                          token=token, force_download=True)
        shutil.rmtree(os.path.join(local_dir, '.cache'))
        print(pretrained_model_name_or_path)
    except Exception as e:
        print(e)
