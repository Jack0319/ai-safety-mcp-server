"""Activation patching/intervention tool for causal analysis.

References:
- "A Mathematical Framework for Transformer Circuits" (Elhage et al., 2021)
  https://transformer-circuits.pub/2021/framework/index.html
- "In-context Learning and Induction Heads" (Olsson et al., 2022)
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- TransformerLens activation patching: https://github.com/neelnanda-io/TransformerLens
- Anthropic's activation patching methodology
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from core.errors import UnsupportedModelError
from safety_interp.backend import InterpretabilityBackend


class ActivationPatchingTool:
    """Tool for patching activations to test causal effects."""

    def __init__(self, backend: InterpretabilityBackend):
        self.backend = backend

    def run(
        self,
        *,
        input_text: str,
        patch_text: str,
        layer_index: int,
        patch_position: Optional[int] = None,
        patch_type: str = "residual",  # "residual", "mlp", "attn"
    ) -> Dict[str, object]:
        """
        Patch activations from patch_text into input_text at specified layer.

        Args:
            input_text: Base text to patch into
            patch_text: Source text to extract activations from
            layer_index: Layer to patch at
            patch_position: Position to patch (None = last position)
            patch_type: Type of activation to patch ("residual", "mlp", "attn")
        """
        layer_index = self.backend.ensure_layer_index(layer_index)

        if self.backend.use_stub:
            return {
                "original_output": "stub",
                "patched_output": "stub",
                "effect_size": 0.0,
                "layer_index": layer_index,
                "patch_type": patch_type,
            }

        tokenizer = self.backend.tokenizer
        model = self.backend.model
        model.eval()

        # Encode both texts
        input_encoded = tokenizer(input_text, return_tensors="pt")
        patch_encoded = tokenizer(patch_text, return_tensors="pt")

        with torch.no_grad():
            # Get original output
            original_outputs = model(**input_encoded, output_hidden_states=True)
            original_logits = original_outputs.logits
            original_next_token = torch.argmax(original_logits[0, -1, :]).item()

            # Get activations from patch text
            patch_outputs = model(**patch_encoded, output_hidden_states=True)
            if patch_type == "residual":
                patched_activations = patch_outputs.hidden_states[layer_index].clone()
            else:
                raise NotImplementedError(f"Patch type {patch_type} not yet implemented")

            # Create hook to patch activations
            def patch_hook(module, input, output):
                if patch_type == "residual":
                    # Patch residual stream - output is typically a tuple (hidden_states, ...)
                    if isinstance(output, tuple):
                        hidden = output[0].clone()
                        if patch_position is not None:
                            hidden[0, patch_position, :] = patched_activations[0, patch_position, :]
                        else:
                            # Patch last position
                            seq_len = hidden.shape[1]
                            patch_seq_len = patched_activations.shape[1]
                            hidden[0, -1, :] = patched_activations[0, min(patch_seq_len - 1, seq_len - 1), :]
                        return (hidden,) + output[1:]
                    else:
                        # Direct tensor output
                        if patch_position is not None:
                            output[0, patch_position, :] = patched_activations[0, patch_position, :]
                        else:
                            seq_len = output.shape[1]
                            patch_seq_len = patched_activations.shape[1]
                            output[0, -1, :] = patched_activations[0, min(patch_seq_len - 1, seq_len - 1), :]
                return output

            # Register hook on the appropriate layer
            # Try common model architectures
            if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layer = model.transformer.h[layer_index]
            elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
                layer = model.gpt_neox.layers[layer_index]
            elif hasattr(model, "model") and hasattr(model.model, "layers"):
                layer = model.model.layers[layer_index]
            else:
                raise UnsupportedModelError("Cannot find transformer layers in model")

            handle = layer.register_forward_hook(patch_hook)

            try:
                # Run with patched activations
                patched_outputs = model(**input_encoded, output_hidden_states=True)
                patched_logits = patched_outputs.logits
                patched_next_token = torch.argmax(patched_logits[0, -1, :]).item()
            finally:
                handle.remove()

            # Calculate effect size (KL divergence)
            original_probs = F.softmax(original_logits[0, -1, :], dim=-1)
            patched_probs = F.softmax(patched_logits[0, -1, :], dim=-1)
            effect_size = float(F.kl_div(patched_probs.log(), original_probs, reduction="sum"))

            return {
                "original_output": tokenizer.decode([original_next_token]),
                "patched_output": tokenizer.decode([patched_next_token]),
                "effect_size": effect_size,
                "layer_index": layer_index,
                "patch_type": patch_type,
                "original_probs": original_probs.topk(5).values.tolist(),
                "patched_probs": patched_probs.topk(5).values.tolist(),
            }

