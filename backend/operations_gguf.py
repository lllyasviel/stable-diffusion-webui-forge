import gguf
import torch


# def functional_quantize_gguf(weight):
#     gguf_cls = weight.gguf_cls
#     gguf_cls.en


def functional_linear_gguf(x, weight, bias=None):
    target_dtype = x.dtype
    weight = dequantize_tensor(weight).to(target_dtype)
    bias = dequantize_tensor(bias).to(target_dtype)
    return torch.nn.functional.linear(x, weight, bias)


def dequantize_tensor(tensor):
    if tensor is None:
        return None

    data = torch.tensor(tensor.data)
    gguf_cls = tensor.gguf_cls
    gguf_real_shape = tensor.gguf_real_shape

    if gguf_cls is None:
        return data

    return gguf_cls.dequantize_pytorch(data, gguf_real_shape)
