from __future__ import annotations

import pytest

from core.config import EvalConfig, InterpretabilityConfig
from mcp_server.tool_registry import ToolRegistry
from safety_evals import SafetyEvalSuite
from safety_interp import InterpretabilitySuite
from safety_kb.models import Document
from safety_kb.retrieval import KnowledgeBaseClient


@pytest.mark.asyncio
async def test_registry_dispatch_roundtrip():
    kb = KnowledgeBaseClient(vectorstore_url=None, collection="test", docstore_url=None, use_stub=True)
    kb.load_stub_documents(
        [
            Document(doc_id="doc-x", title="Alignment", text="Discusses deception", metadata={"topic": "deception"}),
        ]
    )
    eval_suite = SafetyEvalSuite.from_config(EvalConfig(model="stub", litellm_api_key=None, use_stub=True))
    interp_suite = InterpretabilitySuite.from_config(InterpretabilityConfig(model_dir=None, use_stub=True))
    registry = ToolRegistry(kb=kb, evals=eval_suite, interp=interp_suite)

    tools = await registry.list_tools()
    assert any(tool.name == "kb.search" for tool in tools)

    search_result = await registry.dispatch("kb.search", {"query": "deception"})
    assert search_result["results"]

    eval_result = await registry.dispatch("eval.overall_risk", {"text": "misleading exploit"})
    assert eval_result["overall_label"] in {"low", "medium", "high"}
