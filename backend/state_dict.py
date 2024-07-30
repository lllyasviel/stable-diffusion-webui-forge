import torch


class StateDictItem:
    def __init__(self, key, value, advanced_indexing=None):
        self.key = key
        self.value = value
        self.shape = value.shape
        self.advanced_indexing = advanced_indexing

    def __getitem__(self, advanced_indexing):
        t = self.value[advanced_indexing]
        return StateDictItem(self.key, t, advanced_indexing=advanced_indexing)


def split_state_dict_with_prefix(sd, prefix):
    vae_sd = {}

    for k, v in list(sd.items()):
        if k.startswith(prefix):
            vae_sd[k] = StateDictItem(k[len(prefix):], v)
            del sd[k]

    return vae_sd


def compile_state_dict(state_dict):
    sd = {}
    mapping = {}
    for k, v in state_dict.items():
        sd[k] = v.value
        mapping[v.key] = (k, v.advanced_indexing)
    return sd, mapping


def map_state_dict(sd, mapping):
    new_sd = {}
    for k, v in sd.items():
        k, indexing = mapping.get(k, (k, None))
        if indexing is not None:
            v = v[indexing]
        new_sd[k] = v
    return new_sd


def map_state_dict_heuristic(sd, mapping):
    new_mapping = {}
    for k, (v, _) in mapping:
        new_mapping[k.rpartition('.')[0]] = v.rpartition('.')[0]

    new_sd = {}
    for k, v in sd.items():
        l, m, r = k.rpartition('.')
        l = new_mapping.get(l, l)
        new_sd[l + m + r] = v
    return new_sd
