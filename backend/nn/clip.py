import torch

from transformers import CLIPTextModel, CLIPTextConfig


class IntegratedCLIP(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.transformer = CLIPTextModel(config)
        self.text_projection = torch.nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
