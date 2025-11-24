"""Gradient-based attribution tool.

References:
- "Integrated Gradients" (Sundararajan et al., 2017) - https://arxiv.org/abs/1703.01365
- "Grad-CAM: Visual Explanations from Deep Networks" (Selvaraju et al., 2017)
- "The (Un)reliability of saliency methods" (Adebayo et al., 2018)
- Captum library: https://github.com/pytorch/captum
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from safety_interp.backend import InterpretabilityBackend


class GradientAttributionTool:
    """Tool for computing gradient-based attributions."""

    def __init__(self, backend: InterpretabilityBackend):
        self.backend = backend

    def run(
        self,
        *,
        input_text: str,
        target_token: Optional[str] = None,
        method: str = "integrated_gradients",  # "gradient", "integrated_gradients", "saliency"
    ) -> Dict[str, object]:
        """
        Compute gradient-based attributions.

        Args:
            input_text: Input text
            target_token: Token to attribute to (None = top predicted token)
            method: Attribution method to use
        """
        if self.backend.use_stub:
            tokens = input_text.split()
            return {
                "tokens": tokens,
                "attributions": [0.1] * len(tokens),
                "method": method,
                "summary": "Stub gradient attribution results",
            }

        tokenizer = self.backend.tokenizer
        model = self.backend.model
        model.train()  # Enable gradients

        encoded = tokenizer(input_text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        input_ids.requires_grad = True

        # Get target token
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            if target_token:
                target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
            else:
                target_id = torch.argmax(logits[0, -1, :]).item()

        if method == "gradient":
            # Simple gradient
            outputs = model(input_ids=input_ids)
            target_logit = outputs.logits[0, -1, target_id]
            target_logit.backward()
            attributions = input_ids.grad[0].abs().sum(dim=-1).tolist()

        elif method == "integrated_gradients":
            # Integrated gradients
            baseline = torch.zeros_like(input_ids)
            steps = 50
            attributions = torch.zeros_like(input_ids, dtype=torch.float32)

            for step in range(steps):
                alpha = step / steps
                interpolated = baseline + alpha * (input_ids - baseline)
                interpolated.requires_grad = True

                outputs = model(input_ids=interpolated)
                target_logit = outputs.logits[0, -1, target_id]
                target_logit.backward()

                attributions += interpolated.grad[0].abs()

            attributions = (attributions / steps * (input_ids - baseline)).sum(dim=-1).tolist()

        elif method == "saliency":
            # Saliency map
            outputs = model(input_ids=input_ids)
            target_logit = outputs.logits[0, -1, target_id]
            target_logit.backward()
            attributions = input_ids.grad[0].abs().max(dim=-1)[0].tolist()
        else:
            raise ValueError(f"Unknown method: {method}")

        model.eval()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        return {
            "tokens": tokens,
            "target_token": tokenizer.decode([target_id]),
            "attributions": attributions,
            "method": method,
            "top_attributed": sorted(
                zip(tokens, attributions), key=lambda x: x[1], reverse=True
            )[:5],
            "summary": f"Computed {method} attributions for {len(tokens)} tokens",
        }

