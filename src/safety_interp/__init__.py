"""Interpretability suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.config import InterpretabilityConfig
from safety_interp.activations import summarize_last_layer
from safety_interp.attention import AttentionTool
from safety_interp.backend import InterpretabilityBackend
from safety_interp.logit_lens import LogitLensTool


@dataclass
class InterpretabilitySuite:
    backend: InterpretabilityBackend
    logit_tool: LogitLensTool
    attention_tool: AttentionTool

    @classmethod
    def from_config(cls, config: InterpretabilityConfig) -> "InterpretabilitySuite":
        backend = InterpretabilityBackend(config.model_dir, config.use_stub)
        return cls(backend=backend, logit_tool=LogitLensTool(backend), attention_tool=AttentionTool(backend))

    def logit_lens(self, *, model_id: str, layer_index: int, input_text: str, top_k: int = 10) -> dict[str, object]:
        # model_id kept for future multi-model routing; ensure requested id matches configured path
        if model_id != str(self.backend.model_dir) and not self.backend.use_stub:
            raise ValueError("Requested model_id does not match configured interpretability model")
        return self.logit_tool.run(input_text=input_text, layer_index=layer_index, top_k=top_k)

    def attention(self, *, model_id: str, layer_index: int, head_index: int, input_text: str) -> dict[str, object]:
        if model_id != str(self.backend.model_dir) and not self.backend.use_stub:
            raise ValueError("Requested model_id does not match configured interpretability model")
        return self.attention_tool.run(
            input_text=input_text,
            layer_index=layer_index,
            head_index=head_index,
        )

    def activation_summary(self, *, input_text: str) -> dict[str, object]:
        return summarize_last_layer(input_text, self.backend)


__all__ = ["InterpretabilitySuite"]
