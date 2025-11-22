"""Register MCP tools and dispatch handlers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

from mcp import types

from core.errors import KnowledgeBaseError, MCPSafetyError
from safety_evals import SafetyEvalSuite
from safety_interp import InterpretabilitySuite
from safety_kb.retrieval import KnowledgeBaseClient, KnowledgeBaseFilters
from safety_governance.retrieval import GovernanceRetriever

from . import schemas


@dataclass
class ToolHandler:
    name: str
    description: str
    input_schema: dict
    output_schema: Optional[dict]
    handler: Callable[[Dict], Awaitable[Dict]]


class ToolRegistry:
    def __init__(
        self,
        *,
        kb: KnowledgeBaseClient,
        evals: SafetyEvalSuite,
        interp: InterpretabilitySuite,
        governance: Optional[GovernanceRetriever] = None,
    ) -> None:
        self.kb = kb
        self.evals = evals
        self.interp = interp
        self.governance = governance
        self._tools: Dict[str, ToolHandler] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        self._add_tool(
            ToolHandler(
                name="kb.search",
                description="Semantic search over the AI Safety knowledge base.",
                input_schema=schemas.KB_SEARCH_INPUT,
                output_schema=schemas.KB_SEARCH_OUTPUT,
                handler=self._kb_search,
            )
        )
        self._add_tool(
            ToolHandler(
                name="kb.get_document",
                description="Retrieve a document by doc_id, returning full text.",
                input_schema=schemas.KB_GET_DOCUMENT_INPUT,
                output_schema=schemas.KB_GET_DOCUMENT_OUTPUT,
                handler=self._kb_get_document,
            )
        )
        self._add_tool(
            ToolHandler(
                name="eval.deception",
                description="Estimate deception or obfuscation risk for a text snippet.",
                input_schema=schemas.EVAL_INPUT,
                output_schema=schemas.EVAL_OUTPUT,
                handler=self._eval_deception,
            )
        )
        self._add_tool(
            ToolHandler(
                name="eval.dual_use",
                description="Assess dual-use misuse risk for a text snippet.",
                input_schema=schemas.EVAL_INPUT,
                output_schema=schemas.DUAL_USE_OUTPUT,
                handler=self._eval_dual_use,
            )
        )
        self._add_tool(
            ToolHandler(
                name="eval.capability_risk",
                description="Score dangerous capability development risk.",
                input_schema=schemas.EVAL_INPUT,
                output_schema=schemas.CAPABILITY_OUTPUT,
                handler=self._eval_capability,
            )
        )
        self._add_tool(
            ToolHandler(
                name="eval.overall_risk",
                description="Aggregate deception, dual-use, and capability risk.",
                input_schema=schemas.OVERALL_INPUT,
                output_schema=schemas.OVERALL_OUTPUT,
                handler=self._eval_overall,
            )
        )
        self._add_tool(
            ToolHandler(
                name="interp.logit_lens",
                description="Inspect intermediate layer logits for a causal LM.",
                input_schema=schemas.LOGIT_LENS_INPUT,
                output_schema=None,
                handler=self._interp_logit_lens,
            )
        )
        self._add_tool(
            ToolHandler(
                name="interp.attention",
                description="Inspect attention weights for a given layer/head.",
                input_schema=schemas.ATTENTION_INPUT,
                output_schema=None,
                handler=self._interp_attention,
            )
        )
        self._add_tool(
            ToolHandler(
                name="sys.health_check",
                description="Report backend readiness for KB/Eval/Interp subsystems.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                output_schema=schemas.HEALTH_CHECK_OUTPUT,
                handler=self._health_check,
            )
        )
        if self.governance:
            self._add_tool(
                ToolHandler(
                    name="gov.search",
                    description="Search governance/policy documents.",
                    input_schema=schemas.KB_SEARCH_INPUT,
                    output_schema=schemas.KB_SEARCH_OUTPUT,
                    handler=self._gov_search,
                )
            )

    def _add_tool(self, tool: ToolHandler) -> None:
        self._tools[tool.name] = tool

    async def list_tools(self) -> List[types.Tool]:
        tools: List[types.Tool] = []
        for handler in self._tools.values():
            tools.append(
                types.Tool(
                    name=handler.name,
                    description=handler.description,
                    inputSchema=handler.input_schema,
                    outputSchema=handler.output_schema,
                )
            )
        return tools

    async def dispatch(self, name: str, arguments: Dict) -> Dict:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' is not registered")
        handler = self._tools[name].handler
        return await handler(arguments)

    async def _kb_search(self, params: Dict) -> Dict:
        filters = params.get("filters") or {}
        kb_filters = KnowledgeBaseFilters(
            topic=filters.get("topic"),
            org=filters.get("org"),
            year_min=filters.get("year_min"),
            year_max=filters.get("year_max"),
        )
        results = self.kb.search(params["query"], k=params.get("k", 5), filters=kb_filters)
        return {"results": [r.model_dump() for r in results]}

    async def _kb_get_document(self, params: Dict) -> Dict:
        doc = self.kb.get_document(params["doc_id"])
        return doc.model_dump()

    async def _eval_deception(self, params: Dict) -> Dict:
        return self.evals.eval_deception(params["text"], context=params.get("context"))

    async def _eval_dual_use(self, params: Dict) -> Dict:
        return self.evals.eval_dual_use(params["text"], context=params.get("context"))

    async def _eval_capability(self, params: Dict) -> Dict:
        return self.evals.eval_capability_risk(params["text"], context=params.get("context"))

    async def _eval_overall(self, params: Dict) -> Dict:
        return self.evals.eval_overall_risk(params["text"])

    async def _interp_logit_lens(self, params: Dict) -> Dict:
        return self.interp.logit_lens(
            model_id=params["model_id"],
            layer_index=params["layer_index"],
            input_text=params["input_text"],
            top_k=params.get("top_k", 10),
        )

    async def _interp_attention(self, params: Dict) -> Dict:
        return self.interp.attention(
            model_id=params["model_id"],
            layer_index=params["layer_index"],
            head_index=params["head_index"],
            input_text=params["input_text"],
        )

    async def _health_check(self, _: Dict) -> Dict:
        kb_status = "stub" if self.kb.stub else "ready"
        eval_status = "stub" if self.evals.client.use_stub else "ready"
        interp_status = "stub" if self.interp.backend.use_stub else "ready"
        return {"status": "ok", "kb": kb_status, "evals": eval_status, "interp": interp_status}

    async def _gov_search(self, params: Dict) -> Dict:
        if not self.governance:
            raise KnowledgeBaseError(code="GOV_DISABLED", message="Governance corpus not configured")
        results = self.governance.search(params["query"], k=params.get("k", 5))
        return {"results": [r.model_dump() for r in results]}
