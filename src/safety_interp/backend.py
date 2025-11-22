"""Shared interpretability backend utilities."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.errors import UnsupportedModelError


class InterpretabilityBackend:
    def __init__(self, model_dir: Optional[Path], use_stub: bool = False):
        self.model_dir = Path(model_dir) if model_dir else None
        self.use_stub = use_stub or self.model_dir is None
        self._artifacts: Tuple | None = None
        if not self.use_stub and self.model_dir:
            self._artifacts = _load_artifacts(str(self.model_dir))

    @property
    def tokenizer(self):  # type: ignore[override]
        if not self._artifacts:
            raise UnsupportedModelError(str(self.model_dir))
        return self._artifacts[0]

    @property
    def model(self):  # type: ignore[override]
        if not self._artifacts:
            raise UnsupportedModelError(str(self.model_dir))
        return self._artifacts[1]

    def ensure_layer_index(self, layer_index: int) -> int:
        if self.use_stub:
            return max(0, layer_index)
        num_layers = self.model.config.num_hidden_layers
        if layer_index < 0 or layer_index >= num_layers:
            raise UnsupportedModelError(f"Layer index {layer_index} out of range (0-{num_layers - 1})")
        return layer_index


@lru_cache(maxsize=2)
def _load_artifacts(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True, output_attentions=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model
