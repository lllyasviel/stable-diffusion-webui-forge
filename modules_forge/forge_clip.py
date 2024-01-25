from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWords


class CLIP_SD_15_L(FrozenCLIPEmbedderWithCustomWords):
    def encode_with_transformers(self, tokens):
        return super().encode_with_transformers(tokens)


class CLIP_SD_21_G(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        if self.wrapped.layer == "penultimate":
            self.wrapped.layer = "hidden"
            self.wrapped.layer_idx = -2

    def encode_with_transformers(self, tokens):
        return super().encode_with_transformers(tokens)


class CLIP_SD_XL_L(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=self.wrapped.layer == "hidden")

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

    def encode_with_transformers(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=self.wrapped.layer == "hidden")

        if self.wrapped.layer == "last":
            z = outputs.last_hidden_state
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]

        z.pooled = outputs.pooler_output

        return z
