import torch


def load_state_dict(model, sd, ignore_errors=[], log_name=None, ignore_start=None):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [x for x in missing if x not in ignore_errors]
    unexpected = [x for x in unexpected if x not in ignore_errors]

    if isinstance(ignore_start, str):
        missing = [x for x in missing if not x.startswith(ignore_start)]
        unexpected = [x for x in unexpected if not x.startswith(ignore_start)]

    log_name = log_name or type(model).__name__
    if len(missing) > 0:
        print(f'{log_name} Missing: {missing}')
    if len(unexpected) > 0:
        print(f'{log_name} Unexpected: {unexpected}')
    return


def state_dict_has(sd, prefix):
    return any(x.startswith(prefix) for x in sd.keys())


def filter_state_dict_with_prefix(sd, prefix, new_prefix=''):
    new_sd = {}

    for k, v in list(sd.items()):
        if k.startswith(prefix):
            new_sd[new_prefix + k[len(prefix):]] = v
            del sd[k]

    return new_sd


def try_filter_state_dict(sd, prefix_list, new_prefix=''):
    for prefix in prefix_list:
        if state_dict_has(sd, prefix):
            return filter_state_dict_with_prefix(sd, prefix, new_prefix)
    return {}


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]
    return sd


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out
