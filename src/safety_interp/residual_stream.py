"""Residual stream analysis tool.

References:
- "A Mathematical Framework for Transformer Circuits" (Elhage et al., 2021)
  https://transformer-circuits.pub/2021/framework/index.html
- Residual stream concept from Anthropic's mechanistic interpretability research
- TransformerLens residual stream utilities
"""
from __future__ import annotations

from typing import Dict, List

import torch

from safety_interp.backend import InterpretabilityBackend


class ResidualStreamTool:
    """Tool for analyzing the residual stream."""

    def __init__(self, backend: InterpretabilityBackend):
        self.backend = backend

    def run(
        self,
        *,
        input_text: str,
        layer_index: int,
        analysis_type: str = "statistics",  # "statistics", "norm", "cosine_similarity"
    ) -> Dict[str, object]:
        """
        Analyze residual stream at a specific layer.

        Args:
            input_text: Input text
            layer_index: Layer to analyze
            analysis_type: Type of analysis to perform
        """
        layer_index = self.backend.ensure_layer_index(layer_index)

        if self.backend.use_stub:
            tokens = input_text.split()
            return {
                "tokens": tokens,
                "statistics": {"mean": 0.0, "std": 1.0, "norm": float(len(tokens))},
                "layer_index": layer_index,
                "analysis_type": analysis_type,
            }

        tokenizer = self.backend.tokenizer
        model = self.backend.model
        model.eval()

        encoded = tokenizer(input_text, return_tensors="pt")
        seq_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            if layer_index >= len(hidden_states):
                raise ValueError(f"Layer {layer_index} out of range")

            residual = hidden_states[layer_index][0]  # [seq_len, hidden_dim]

            if analysis_type == "statistics":
                mean = residual.mean(dim=-1).tolist()
                std = residual.std(dim=-1).tolist()
                norm = torch.norm(residual, dim=-1).tolist()
                stats = {
                    "mean": mean,
                    "std": std,
                    "norm": norm,
                    "overall_mean": float(residual.mean().item()),
                    "overall_std": float(residual.std().item()),
                }
            elif analysis_type == "norm":
                norms = torch.norm(residual, dim=-1).tolist()
                stats = {"norms": norms, "max_norm": float(max(norms)), "min_norm": float(min(norms))}
            elif analysis_type == "cosine_similarity":
                # Compute cosine similarity between consecutive tokens
                if seq_len < 2:
                    similarities = []
                else:
                    normalized = F.normalize(residual, p=2, dim=-1)
                    similarities = [
                        float(F.cosine_similarity(normalized[i : i + 1], normalized[i + 1 : i + 2]).item())
                        for i in range(seq_len - 1)
                    ]
                stats = {"cosine_similarities": similarities, "avg_similarity": float(sum(similarities) / len(similarities)) if similarities else 0.0}
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")

        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        return {
            "tokens": tokens,
            "statistics": stats,
            "layer_index": layer_index,
            "analysis_type": analysis_type,
            "hidden_dim": residual.shape[-1],
            "sequence_length": seq_len,
        }

