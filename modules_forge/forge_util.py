import torch
import numpy as np

from ldm_patched.modules.conds import CONDRegular, CONDCrossAttn


def cond_from_a1111_to_patched_ldm(cond):
    if isinstance(cond, torch.Tensor):
        result = dict(
            cross_attn=cond,
            model_conds=dict(
                c_crossattn=CONDCrossAttn(cond),
            )
        )
        return [result, ]

    cross_attn = cond['crossattn']
    pooled_output = cond['vector']

    result = dict(
        cross_attn=cross_attn,
        pooled_output=pooled_output,
        model_conds=dict(
            c_crossattn=CONDCrossAttn(cross_attn),
            y=CONDRegular(pooled_output)
        )
    )

    return [result, ]


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y
