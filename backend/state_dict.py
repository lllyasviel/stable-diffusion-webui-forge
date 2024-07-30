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


def shrink_last_key(t):
    ts = t.split('.')
    del ts[-1]
    return '.'.join(ts)


def compile_state_dict(state_dict):
    sd = {}
    mapping = {}
    for k, v in state_dict.items():
        sd[k] = v.value
        mapping[shrink_last_key(v.key)] = shrink_last_key(k)
    return sd, mapping
