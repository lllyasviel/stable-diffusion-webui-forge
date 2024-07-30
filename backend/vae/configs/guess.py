import os
import json

dir_path = os.path.dirname(__file__)


def guess_vae_config(state_dict):
    p = os.path.join(dir_path, 'sd15.json')
    return json.load(open(p, encoding='utf-8'))
