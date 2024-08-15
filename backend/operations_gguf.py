import gguf
import torch


quants_mapping = {
    gguf.GGMLQuantizationType.Q4_0: gguf.Q4_0,
    gguf.GGMLQuantizationType.Q5_0: gguf.Q5_0,
    gguf.GGMLQuantizationType.Q8_0: gguf.Q8_0,
}


def functional_linear_gguf(x, weight, bias=None):
    target_dtype = x.dtype
    weight = dequantize_tensor(weight, target_dtype)
    bias = dequantize_tensor(bias, target_dtype)
    return torch.nn.functional.linear(x, weight, bias)


def dequantize_tensor(tensor, target_dtype=torch.float16):
    if tensor is None:
        return None

    data = torch.tensor(tensor.data)
    gguf_type = tensor.gguf_type
    gguf_real_shape = tensor.gguf_real_shape

    if gguf_type in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16]:
        return data.to(target_dtype)

    if gguf_type not in quants_mapping:
        raise NotImplementedError(f'Quant type {gguf_type} not implemented!')

    quant_cls = quants_mapping.get(gguf_type)

    return quant_cls.dequantize_pytorch(data, gguf_real_shape).to(target_dtype)
