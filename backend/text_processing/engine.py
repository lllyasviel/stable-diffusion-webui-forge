import math
from collections import namedtuple

import torch

from backend.text_processing import parsing, emphasis


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


class ClassicTextProcessingEngine(torch.nn.Module):
    def __init__(self, wrapped, hijack):
        super().__init__()
        self.chunk_length = 75

        self.is_trainable = False
        self.input_key = 'txt'
        self.return_pooled = False

        self.comma_token = None

        self.hijack = hijack

        self.wrapped = wrapped

        self.is_trainable = getattr(wrapped, 'is_trainable', False)
        self.input_key = getattr(wrapped, 'input_key', 'txt')
        self.return_pooled = getattr(self.wrapped, 'return_pooled', False)

        self.legacy_ucg_val = None  # for sgm codebase

        self.tokenizer = wrapped.tokenizer

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(',</w>', None)

        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

        self.id_start = self.wrapped.tokenizer.bos_token_id
        self.id_end = self.wrapped.tokenizer.eos_token_id
        self.id_pad = self.id_end

    def empty_chunk(self):
        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        tokenized = self.wrapped.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        return tokenized

    def encode_with_transformers(self, tokens):
        raise NotImplementedError

    def encode_embedding_init_text(self, init_text, nvpt):
        embedding_layer = self.wrapped.transformer.text_model.embeddings
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        embedded = embedding_layer.token_embedding.wrapped(ids.to(embedding_layer.token_embedding.wrapped.weight.device)).squeeze(0)
        return embedded

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                comma_padding_backtrack = 20

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                elif comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, position)
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue

                emb_len = int(embedding.vectors)
                if len(chunk.tokens) + emb_len > self.chunk_length:
                    next_chunk()

                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

                chunk.tokens += [0] * emb_len
                chunk.multipliers += [weight] * emb_len
                position += embedding_length_in_tokens

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts):
        token_count = 0

        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def forward(self, texts):
        batch_chunks, token_count = self.process_texts(texts)

        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.hijack.fixes = [x.fixes for x in batch_chunk]

            for fixes in self.hijack.fixes:
                for _position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        if used_embeddings:
            for name, embedding in used_embeddings.items():
                print(f'Used Embedding: {name}')

        # Todo:
        # if opts.textual_inversion_add_hashes_to_infotext and used_embeddings:
        #     hashes = []
        #     for name, embedding in used_embeddings.items():
        #         shorthash = embedding.shorthash
        #         if not shorthash:
        #             continue
        #
        #         name = name.replace(":", "").replace(",", "")
        #         hashes.append(f"{name}: {shorthash}")
        #
        #     if hashes:
        #         if self.hijack.extra_generation_params.get("TI hashes"):
        #             hashes.append(self.hijack.extra_generation_params.get("TI hashes"))
        #         self.hijack.extra_generation_params["TI hashes"] = ", ".join(hashes)
        #
        # if any(x for x in texts if "(" in x or "[" in x) and opts.emphasis != "Original":
        #     self.hijack.extra_generation_params["Emphasis"] = opts.emphasis

        if self.return_pooled:
            return torch.hstack(zs), zs[0].pooled
        else:
            return torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        tokens = torch.asarray(remade_batch_tokens)

        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1:tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        pooled = getattr(z, 'pooled', None)

        # Todo
        # e = emphasis.get_current_option(opts.emphasis)()

        e = emphasis.EmphasisOriginal()
        e.tokens = remade_batch_tokens
        e.multipliers = torch.asarray(batch_multipliers)
        e.z = z
        e.after_transformers()
        z = e.z

        if pooled is not None:
            z.pooled = pooled

        return z
