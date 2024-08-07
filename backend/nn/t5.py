import torch
import math

from backend.attention import attention_function


activations = {
    "gelu_pytorch_tanh": lambda a: torch.nn.functional.gelu(a, approximate="tanh"),
    "relu": torch.nn.functional.relu,
}


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(x) * x


class T5DenseActDense(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, ff_activation):
        super().__init__()
        self.wi = torch.nn.Linear(model_dim, ff_dim, bias=False)
        self.wo = torch.nn.Linear(ff_dim, model_dim, bias=False)
        self.act = activations[ff_activation]

    def forward(self, x):
        x = self.act(self.wi(x))
        x = self.wo(x)
        return x


class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, ff_activation):
        super().__init__()
        self.wi_0 = torch.nn.Linear(model_dim, ff_dim, bias=False)
        self.wi_1 = torch.nn.Linear(model_dim, ff_dim, bias=False)
        self.wo = torch.nn.Linear(ff_dim, model_dim, bias=False)
        self.act = activations[ff_activation]

    def forward(self, x):
        hidden_gelu = self.act(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, ff_activation, gated_act):
        super().__init__()
        if gated_act:
            self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, ff_activation)
        else:
            self.DenseReluDense = T5DenseActDense(model_dim, ff_dim, ff_activation)

        self.layer_norm = T5LayerNorm(model_dim)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias):
        super().__init__()
        self.q = torch.nn.Linear(model_dim, inner_dim, bias=False)
        self.k = torch.nn.Linear(model_dim, inner_dim, bias=False)
        self.v = torch.nn.Linear(model_dim, inner_dim, bias=False)
        self.o = torch.nn.Linear(inner_dim, model_dim, bias=False)
        self.num_heads = num_heads

        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = torch.nn.Embedding(self.relative_attention_num_buckets, self.num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device, dtype):
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket).to(dtype)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, x, mask=None, past_bias=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device, x.dtype)

        if past_bias is not None:
            if mask is not None:
                mask = mask + past_bias
            else:
                mask = past_bias

        out = attention_function(q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask)
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias)
        self.layer_norm = T5LayerNorm(model_dim)

    def forward(self, x, mask=None, past_bias=None):
        output, past_bias = self.SelfAttention(self.layer_norm(x), mask=mask, past_bias=past_bias)
        x += output
        return x, past_bias


class T5Block(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, ff_activation, gated_act, num_heads, relative_attention_bias):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias))
        self.layer.append(T5LayerFF(model_dim, ff_dim, ff_activation, gated_act))

    def forward(self, x, mask=None, past_bias=None):
        x, past_bias = self.layer[0](x, mask, past_bias)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):
    def __init__(self, num_layers, model_dim, inner_dim, ff_dim, ff_activation, gated_act, num_heads, relative_attention):
        super().__init__()

        self.block = torch.nn.ModuleList(
            [T5Block(model_dim, inner_dim, ff_dim, ff_activation, gated_act, num_heads, relative_attention_bias=((not relative_attention) or (i == 0))) for i in range(num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(model_dim)

    def forward(self, x, attention_mask=None):
        mask = None

        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), float("-inf"))

        past_bias = None

        for i, l in enumerate(self.block):
            x, past_bias = l(x, mask, past_bias)

        x = self.final_layer_norm(x)
        return x


class T5(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config["num_layers"]
        model_dim = config["d_model"]

        self.encoder = T5Stack(self.num_layers, model_dim, model_dim, config["d_ff"], config["dense_act_fn"], config["is_gated_act"], config["num_heads"], config["model_type"] != "umt5")
        self.shared = torch.nn.Embedding(config["vocab_size"], model_dim)

    def forward(self, input_ids, *args, **kwargs):
        x = self.shared(input_ids)
        x = torch.nan_to_num(x)
        return self.encoder(x, *args, **kwargs)


class IntegratedT5(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = T5(config)
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
