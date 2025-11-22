from __future__ import annotations

from core.config import InterpretabilityConfig
from safety_interp import InterpretabilitySuite


def stub_suite() -> InterpretabilitySuite:
    cfg = InterpretabilityConfig(model_dir=None, use_stub=True)
    return InterpretabilitySuite.from_config(cfg)


def test_logit_lens_stub_returns_tokens():
    suite = stub_suite()
    result = suite.logit_lens(model_id="stub", layer_index=0, input_text="safety alignment", top_k=3)
    assert "tokens" in result
    assert len(result["top_predictions"]) <= 3


def test_attention_stub_uniform_distribution():
    suite = stub_suite()
    output = suite.attention(model_id="stub", layer_index=0, head_index=0, input_text="alignment research")
    weights = output["weights"]
    assert abs(sum(weights) - 1.0) < 1e-6 or not weights
