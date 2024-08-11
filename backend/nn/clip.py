import torch


class IntegratedCLIP(torch.nn.Module):
    def __init__(self, cls, config, add_text_projection=False):
        super().__init__()
        self.transformer = cls(config)
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))

        if add_text_projection:
            embed_dim = config.hidden_size
            self.transformer.text_projection = torch.nn.Linear(embed_dim, embed_dim, bias=False)
