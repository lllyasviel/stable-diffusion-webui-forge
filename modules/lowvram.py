import torch
from modules import devices, shared

module_in_gpu = None
cpu = torch.device("cpu")


def send_everything_to_cpu():
    return


def is_needed(sd_model):
    return False


def apply(sd_model):
    return


def setup_for_low_vram(sd_model, use_medvram):
    return


def is_enabled(sd_model):
    return False
