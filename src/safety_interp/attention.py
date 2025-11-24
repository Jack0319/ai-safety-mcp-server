"""Attention inspection tool.

References:
- "Attention is All You Need" (Vaswani et al., 2017) - Original attention mechanism
- "What Does BERT Look At?" (Clark et al., 2019) - Attention analysis methodology
- TransformerLens: https://github.com/neelnanda-io/TransformerLens
"""
from __future__ import annotations

from typing import Dict

import torch

from core.errors import UnsupportedModelError
from safety_interp.backend import InterpretabilityBackend


class AttentionTool:
    def __init__(self, backend: InterpretabilityBackend):
        self.backend = backend

    def run(
        self,
        *,
        input_text: str,
        layer_index: int,
        head_index: int = 0,
        token_position: int = -1,
    ) -> Dict[str, object]:
        layer_index = self.backend.ensure_layer_index(layer_index)
        if self.backend.use_stub:
            tokens = input_text.split()
            weights = [round(1 / max(1, len(tokens)), 4) for _ in tokens]
            return {
                "tokens": tokens,
                "weights": weights,
                "head_index": head_index,
                "summary": "Stub attention weights (uniform).",
            }

        tokenizer = self.backend.tokenizer
        model = self.backend.model
        encoded = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)
            attentions = outputs.attentions
            if attentions is None:
                raise UnsupportedModelError("Model does not expose attention weights")
            layer_attn = attentions[layer_index]
            if head_index >= layer_attn.shape[1]:
                raise UnsupportedModelError(f"Head index {head_index} out of range")
            seq_len = layer_attn.shape[-1]
            if token_position < 0:
                token_position = seq_len - 1
            weights_tensor = layer_attn[0, head_index, token_position, :]
            weights = weights_tensor.tolist()

        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        summary = {
            "max_token": tokens[int(torch.argmax(weights_tensor))],
            "max_weight": float(torch.max(weights_tensor)),
        }
        return {
            "tokens": tokens,
            "weights": weights,
            "head_index": head_index,
            "summary": summary,
        }
