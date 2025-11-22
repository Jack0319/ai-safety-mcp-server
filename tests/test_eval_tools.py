from __future__ import annotations

from core.config import EvalConfig
from safety_evals import SafetyEvalSuite


def stub_suite() -> SafetyEvalSuite:
    cfg = EvalConfig(model="stub", litellm_api_key=None, use_stub=True)
    return SafetyEvalSuite.from_config(cfg)


def test_deception_eval_stub_scores():
    suite = stub_suite()
    result = suite.eval_deception("This plan hides a backdoor and misleads operators.")
    assert 0.0 <= result["score"] <= 1.0
    assert result["label"] in {"low", "medium", "high"}


def test_overall_risk_combines_components():
    suite = stub_suite()
    output = suite.eval_overall_risk("Autonomous malware that weaponizes exploits")
    assert set(output["components"].keys()) == {"deception", "dual_use", "capability"}
