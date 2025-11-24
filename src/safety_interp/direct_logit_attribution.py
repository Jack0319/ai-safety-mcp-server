"""Direct Logit Attribution (DLA) tool.

References:
- "A Mathematical Framework for Transformer Circuits" (Elhage et al., 2021)
  https://transformer-circuits.pub/2021/framework/index.html
- "In-context Learning and Induction Heads" (Olsson et al., 2022)
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Direct logit attribution methodology from Anthropic's interpretability research
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from safety_interp.backend import InterpretabilityBackend


class DirectLogitAttributionTool:
    """Tool for direct logit attribution to identify important components."""

    def __init__(self, backend: InterpretabilityBackend):
        self.backend = backend

    def run(
        self,
        *,
        input_text: str,
        target_token: Optional[str] = None,
        layer_index: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Compute direct logit attribution from a specific layer.

        Args:
            input_text: Input text
            target_token: Token to attribute to (None = top predicted token)
            layer_index: Layer to compute attribution from (None = all layers)
        """
        if self.backend.use_stub:
            tokens = input_text.split()
            return {
                "tokens": tokens,
                "attributions": [0.1] * len(tokens),
                "top_contributors": tokens[:3],
                "summary": "Stub DLA results",
            }

        tokenizer = self.backend.tokenizer
        model = self.backend.model
        model.eval()

        encoded = tokenizer(input_text, return_tensors="pt")
        seq_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            logits = outputs.logits
            hidden_states = outputs.hidden_states

            if target_token:
                target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
            else:
                target_id = torch.argmax(logits[0, -1, :]).item()

            target_logit = logits[0, -1, target_id].item()

            # Get output embeddings
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is None:
                raise ValueError("Model does not have output embeddings")

            # Compute attributions
            if layer_index is not None:
                layer_index = self.backend.ensure_layer_index(layer_index)
                hidden = hidden_states[layer_index][0, -1, :]  # Last token, last layer
                # Project to logit space
                projected = output_embeddings(hidden.unsqueeze(0))
                attribution = projected[0, target_id].item()
                attributions = [attribution]
                layers_analyzed = [layer_index]
            else:
                # Analyze all layers
                attributions = []
                layers_analyzed = []
                for idx, hidden in enumerate(hidden_states):
                    hidden_vec = hidden[0, -1, :]
                    projected = output_embeddings(hidden_vec.unsqueeze(0))
                    attr = projected[0, target_id].item()
                    attributions.append(attr)
                    layers_analyzed.append(idx)

        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        top_contributors = sorted(
            zip(layers_analyzed, attributions), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        return {
            "tokens": tokens,
            "target_token": tokenizer.decode([target_id]),
            "target_logit": target_logit,
            "attributions": attributions,
            "layers_analyzed": layers_analyzed,
            "top_contributors": [{"layer": l, "attribution": a} for l, a in top_contributors],
            "summary": f"Computed DLA for {len(attributions)} layer(s)",
        }

