"""Backend module for stable-diffusion-webui-forge"""

from .attention import get_attn_precision, efficient_dot_product_attention

__all__ = [
    'get_attn_precision',
    'efficient_dot_product_attention',
]
