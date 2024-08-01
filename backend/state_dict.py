import torch


def filter_state_dict_with_prefix(sd, prefix):
    new_sd = {}

    for k, v in list(sd.items()):
        if k.startswith(prefix):
            new_sd[k[len(prefix):]] = v
            del sd[k]

    return new_sd
