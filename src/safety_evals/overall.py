"""Aggregate overall risk evaluation."""
from __future__ import annotations

from typing import Any, Dict

from safety_evals import capability_risk, deception, dual_use
from safety_evals.base import EvalModelClient, score_to_label


def evaluate(text: str, *, client: EvalModelClient) -> Dict[str, Any]:
    deception_result = deception.evaluate_score(text, client=client)
    dual_use_result = dual_use.evaluate_score(text, client=client)
    capability_result = capability_risk.evaluate_score(text, client=client)

    components = {
        "deception": deception_result.score,
        "dual_use": dual_use_result.score,
        "capability": capability_result.score,
    }
    overall_score = sum(components.values()) / len(components)
    label = score_to_label(overall_score)
    explanation = " / ".join(
        filter(
            None,
            [
                deception_result.explanation,
                dual_use_result.explanation,
                capability_result.explanation,
            ],
        )
    ) or "Aggregated risk across deception, dual-use, and capability dimensions."
    return {
        "overall_label": label,
        "components": components,
        "explanation": explanation,
    }
