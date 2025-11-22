"""Logit lens inspection tool."""
from __future__ import annotations

from typing import Dict, List

import torch

from core.errors import UnsupportedModelError
from safety_interp.backend import InterpretabilityBackend


class LogitLensTool:
    def __init__(self, backend: InterpretabilityBackend):
        self.backend = backend

    def run(self, *, input_text: str, layer_index: int, top_k: int = 10) -> dict[str, object]:
        layer_index = self.backend.ensure_layer_index(layer_index)
        if self.backend.use_stub:
            tokens = input_text.split()
            predictions = [
                {"token": token[::-1], "logit": 0.0, "prob": round(1 / max(1, len(tokens)), 4)}
                for token in tokens[:top_k]
            ] or [{"token": "<empty>", "logit": 0.0, "prob": 1.0}]
            return {"tokens": tokens, "top_predictions": predictions[:top_k]}

        tokenizer = self.backend.tokenizer
        model = self.backend.model
        encoded = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise UnsupportedModelError("Model does not expose hidden states")
            chosen_state = hidden_states[layer_index][:, -1, :]
            output_layer = model.get_output_embeddings()
            if output_layer is None:
                raise UnsupportedModelError("Model missing output embeddings for logit lens")
            logits = output_layer(chosen_state)
            logits = logits.squeeze(0)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk = torch.topk(logits, k=top_k)

        vocab_ids = topk.indices.tolist()
        top_predictions = []
        for idx in vocab_ids:
            token = tokenizer.convert_ids_to_tokens(idx)
            logit_value = float(logits[idx])
            prob_value = float(probs[idx])
            top_predictions.append({"token": token, "logit": logit_value, "prob": prob_value})

        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        return {"tokens": tokens, "top_predictions": top_predictions}
