import torch

from transformers import CLIPTextModel, CLIPTextConfig


class IntegratedCLIP(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.transformer = CLIPTextModel(config)
        embed_dim = config.hidden_size
        self.transformer.text_projection = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.transformer.text_projection.weight.copy_(torch.eye(embed_dim))
