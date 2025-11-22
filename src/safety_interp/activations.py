"""Placeholder activation hooks."""
from __future__ import annotations

from typing import Dict

from safety_interp.backend import InterpretabilityBackend


def summarize_last_layer(input_text: str, backend: InterpretabilityBackend) -> Dict[str, object]:
    """Return simple activation statistics for the final hidden layer (stub-friendly)."""

    if backend.use_stub:
        length = len(input_text.split())
        return {"mean": 0.0, "std": 0.0, "tokens": length}

    tokenizer = backend.tokenizer
    model = backend.model
    encoded = tokenizer(input_text, return_tensors="pt")
    outputs = model(**encoded, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model missing hidden states")
    last = hidden_states[-1]
    mean = float(last.mean().item())
    std = float(last.std().item())
    return {"mean": mean, "std": std, "tokens": len(encoded["input_ids"][0])}
