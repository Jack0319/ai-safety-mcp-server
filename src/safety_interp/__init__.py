"""Interpretability suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.config import InterpretabilityConfig
from core.logging import get_logger
from safety_interp.activations import summarize_last_layer
from safety_interp.activation_patching import ActivationPatchingTool
from safety_interp.attention import AttentionTool
from safety_interp.backend import InterpretabilityBackend
from safety_interp.causal_tracing import CausalTracingTool
from safety_interp.direct_logit_attribution import DirectLogitAttributionTool
from safety_interp.gradient_attribution import GradientAttributionTool
from safety_interp.logit_lens import LogitLensTool
from safety_interp.residual_stream import ResidualStreamTool

logger = get_logger(__name__)


@dataclass
class InterpretabilitySuite:
    backend: InterpretabilityBackend
    logit_tool: LogitLensTool
    attention_tool: AttentionTool
    activation_patching_tool: ActivationPatchingTool
    causal_tracing_tool: CausalTracingTool
    dla_tool: DirectLogitAttributionTool
    residual_stream_tool: ResidualStreamTool
    gradient_attribution_tool: GradientAttributionTool

    @classmethod
    def from_config(cls, config: InterpretabilityConfig) -> "InterpretabilitySuite":
        backend = InterpretabilityBackend(config.model_dir, config.use_stub)
        return cls(
            backend=backend,
            logit_tool=LogitLensTool(backend),
            attention_tool=AttentionTool(backend),
            activation_patching_tool=ActivationPatchingTool(backend),
            causal_tracing_tool=CausalTracingTool(backend),
            dla_tool=DirectLogitAttributionTool(backend),
            residual_stream_tool=ResidualStreamTool(backend),
            gradient_attribution_tool=GradientAttributionTool(backend),
        )

    def _check_model_id(self, model_id: str) -> None:
        """Validate that model_id matches the configured model."""
        if self.backend.use_stub:
            logger.debug(f"Model ID check skipped (stub mode): requested={model_id}")
            return  # Stub mode accepts any model_id
        expected_id = self.backend.model_id or str(self.backend.model_dir or "")
        if model_id != expected_id and model_id != "stub":
            logger.warning(f"Model ID mismatch: requested='{model_id}', configured='{expected_id}'")
            raise ValueError(
                f"Requested model_id '{model_id}' does not match configured model '{expected_id}'. "
                f"Use model_id='{expected_id}' or model_id='stub' for stub mode."
            )
        logger.debug(f"Model ID validated: {model_id}")

    def logit_lens(self, *, model_id: str, layer_index: int, input_text: str, top_k: int = 10) -> dict[str, object]:
        self._check_model_id(model_id)
        return self.logit_tool.run(input_text=input_text, layer_index=layer_index, top_k=top_k)

    def attention(self, *, model_id: str, layer_index: int, head_index: int, input_text: str) -> dict[str, object]:
        self._check_model_id(model_id)
        return self.attention_tool.run(
            input_text=input_text,
            layer_index=layer_index,
            head_index=head_index,
        )

    def activation_summary(self, *, input_text: str) -> dict[str, object]:
        return summarize_last_layer(input_text, self.backend)

    def activation_patching(
        self,
        *,
        model_id: str,
        input_text: str,
        patch_text: str,
        layer_index: int,
        patch_position: Optional[int] = None,
        patch_type: str = "residual",
    ) -> dict[str, object]:
        self._check_model_id(model_id)
        return self.activation_patching_tool.run(
            input_text=input_text,
            patch_text=patch_text,
            layer_index=layer_index,
            patch_position=patch_position,
            patch_type=patch_type,
        )

    def causal_tracing(
        self,
        *,
        model_id: str,
        input_text: str,
        target_token: Optional[str] = None,
        layers_to_trace: Optional[list[int]] = None,
    ) -> dict[str, object]:
        self._check_model_id(model_id)
        return self.causal_tracing_tool.run(
            input_text=input_text,
            target_token=target_token,
            layers_to_trace=layers_to_trace,
        )

    def direct_logit_attribution(
        self,
        *,
        model_id: str,
        input_text: str,
        target_token: Optional[str] = None,
        layer_index: Optional[int] = None,
    ) -> dict[str, object]:
        self._check_model_id(model_id)
        return self.dla_tool.run(
            input_text=input_text,
            target_token=target_token,
            layer_index=layer_index,
        )

    def residual_stream(
        self,
        *,
        model_id: str,
        input_text: str,
        layer_index: int,
        analysis_type: str = "statistics",
    ) -> dict[str, object]:
        self._check_model_id(model_id)
        return self.residual_stream_tool.run(
            input_text=input_text,
            layer_index=layer_index,
            analysis_type=analysis_type,
        )

    def gradient_attribution(
        self,
        *,
        model_id: str,
        input_text: str,
        target_token: Optional[str] = None,
        method: str = "integrated_gradients",
    ) -> dict[str, object]:
        self._check_model_id(model_id)
        return self.gradient_attribution_tool.run(
            input_text=input_text,
            target_token=target_token,
            method=method,
        )


__all__ = ["InterpretabilitySuite"]
