import torch


class JointTokenizer:
    def __init__(self, huggingface_components):
        self.clip_l = huggingface_components.get('tokenizer', None)
        self.clip_g = huggingface_components.get('tokenizer_2', None)


class JointCLIP(torch.nn.Module):
    def __init__(self, huggingface_components):
        super().__init__()
        self.clip_l = huggingface_components.get('text_encoder', None)
        self.clip_g = huggingface_components.get('text_encoder_2', None)
