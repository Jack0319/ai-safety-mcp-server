"""Safety evaluation toolkit entry point."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.config import EvalConfig
from safety_evals import capability_risk, deception, dual_use, overall
from safety_evals.base import EvalModelClient


@dataclass
class SafetyEvalSuite:
    client: EvalModelClient

    @classmethod
    def from_config(cls, config: EvalConfig) -> "SafetyEvalSuite":
        client = EvalModelClient(
            model=config.model,
            api_key=config.litellm_api_key,
            use_stub=config.use_stub,
        )
        return cls(client=client)

    def eval_deception(self, text: str, *, context: Optional[str] = None) -> dict:
        return deception.evaluate(text, context=context, client=self.client)

    def eval_dual_use(self, text: str, *, context: Optional[str] = None) -> dict:
        return dual_use.evaluate(text, context=context, client=self.client)

    def eval_capability_risk(self, text: str, *, context: Optional[str] = None) -> dict:
        return capability_risk.evaluate(text, context=context, client=self.client)

    def eval_overall_risk(self, text: str) -> dict:
        return overall.evaluate(text, client=self.client)


__all__ = ["SafetyEvalSuite"]
