from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWords
from ldm_patched.modules import model_management
from modules import sd_models
from modules.shared import opts


def move_clip_to_gpu():
    if sd_models.model_data.sd_model is None:
        print('Error: CLIP called before SD is loaded!')
        return

    model_management.load_model_gpu(sd_models.model_data.sd_model.forge_objects.clip.patcher)
    return


class CLIP_SD_15_L(FrozenCLIPEmbedderWithCustomWords):
    def encode_with_transformers(self, tokens):
        move_clip_to_gpu()
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)

        if opts.CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        return z


class CLIP_SD_21_H(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        if self.wrapped.layer == "penultimate":
            self.wrapped.layer = "hidden"
            self.wrapped.layer_idx = -2

        self.id_start = 49406
        self.id_end = 49407
        self.id_pad = 0

    def encode_with_transformers(self, tokens):
        move_clip_to_gpu()
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        outputs = self.wrapped.transformer(tokens, output_hidden_states=self.wrapped.layer == "hidden")

        if self.wrapped.layer == "last":
            z = outputs.last_hidden_state
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)

        return z


class CLIP_SD_XL_L(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens):
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        outputs = self.wrapped.transformer(tokens, output_hidden_states=self.wrapped.layer == "hidden")

        if self.wrapped.layer == "last":
            z = outputs.last_hidden_state
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]

        return z


class CLIP_SD_XL_G(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        if self.wrapped.layer == "penultimate":
            self.wrapped.layer = "hidden"
            self.wrapped.layer_idx = -2

        self.id_start = 49406
        self.id_end = 49407
        self.id_pad = 0

    def encode_with_transformers(self, tokens):
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        outputs = self.wrapped.transformer(tokens, output_hidden_states=self.wrapped.layer == "hidden")

        if self.wrapped.layer == "last":
            z = outputs.last_hidden_state
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]

        pooled_output = outputs.pooler_output
        text_projection = self.wrapped.text_projection
        pooled_output = pooled_output.float().to(text_projection.device) @ text_projection.float()
        z.pooled = pooled_output
        return z
