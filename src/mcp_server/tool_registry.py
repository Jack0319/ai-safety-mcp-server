"""Register MCP tools and dispatch handlers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

from mcp import types

from core.errors import KnowledgeBaseError, MCPSafetyError
from core.logging import get_logger
from safety_evals import SafetyEvalSuite
from safety_interp import InterpretabilitySuite
from safety_kb.retrieval import KnowledgeBaseClient, KnowledgeBaseFilters
from safety_governance.retrieval import GovernanceRetriever

from . import schemas

logger = get_logger(__name__)


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
        # Only register KB tools if KB is properly configured
        if self.kb.is_available():
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
                name="eval.jailbreak",
                description="Detect jailbreak attempts designed to bypass safety filters.",
                input_schema=schemas.EVAL_INPUT,
                output_schema=schemas.EVAL_OUTPUT,
                handler=self._eval_jailbreak,
            )
        )
        self._add_tool(
            ToolHandler(
                name="eval.prompt_injection",
                description="Detect prompt injection attempts to manipulate model behavior.",
                input_schema=schemas.EVAL_INPUT,
                output_schema=schemas.EVAL_OUTPUT,
                handler=self._eval_prompt_injection,
            )
        )
        self._add_tool(
            ToolHandler(
                name="eval.bias",
                description="Detect biased or discriminatory content in text.",
                input_schema=schemas.EVAL_INPUT,
                output_schema=schemas.EVAL_OUTPUT,
                handler=self._eval_bias,
            )
        )
        self._add_tool(
            ToolHandler(
                name="eval.toxicity",
                description="Detect toxic, rude, or harmful content in text.",
                input_schema=schemas.EVAL_INPUT,
                output_schema=schemas.EVAL_OUTPUT,
                handler=self._eval_toxicity,
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
                name="interp.activation_patching",
                description="Patch activations to test causal effects (activation intervention).",
                input_schema=schemas.ACTIVATION_PATCHING_INPUT,
                output_schema=None,
                handler=self._interp_activation_patching,
            )
        )
        self._add_tool(
            ToolHandler(
                name="interp.causal_tracing",
                description="Trace causal effects through the model by corrupting and restoring activations.",
                input_schema=schemas.CAUSAL_TRACING_INPUT,
                output_schema=None,
                handler=self._interp_causal_tracing,
            )
        )
        self._add_tool(
            ToolHandler(
                name="interp.direct_logit_attribution",
                description="Compute direct logit attribution from specific layers (DLA).",
                input_schema=schemas.DIRECT_LOGIT_ATTRIBUTION_INPUT,
                output_schema=None,
                handler=self._interp_direct_logit_attribution,
            )
        )
        self._add_tool(
            ToolHandler(
                name="interp.residual_stream",
                description="Analyze residual stream statistics, norms, or similarities at a specific layer.",
                input_schema=schemas.RESIDUAL_STREAM_INPUT,
                output_schema=None,
                handler=self._interp_residual_stream,
            )
        )
        self._add_tool(
            ToolHandler(
                name="interp.gradient_attribution",
                description="Compute gradient-based attributions (gradient, integrated gradients, saliency).",
                input_schema=schemas.GRADIENT_ATTRIBUTION_INPUT,
                output_schema=None,
                handler=self._interp_gradient_attribution,
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
        logger.debug(f"Registered tool: {tool.name}")

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
        logger.debug(f"Listed {len(tools)} tools")
        return tools

    async def dispatch(self, name: str, arguments: Dict) -> Dict:
        if name not in self._tools:
            logger.warning(f"Tool '{name}' not found in registry (available: {list(self._tools.keys())})")
            raise ValueError(f"Tool '{name}' is not registered")
        handler = self._tools[name].handler
        logger.debug(f"Dispatching tool '{name}' with {len(arguments)} arguments")
        return await handler(arguments)

    async def _kb_search(self, params: Dict) -> Dict:
        query = params.get("query", "")
        k = params.get("k", 5)
        logger.info(f"KB search: query='{query[:50]}...', k={k}")
        
        if not self.kb.is_available():
            logger.warning("KB search requested but KB not configured")
            raise KnowledgeBaseError(
                code="KB_NOT_CONFIGURED",
                message="Knowledge base is not configured. Please set KB_VECTORSTORE_URL in your environment.",
            )
        filters = params.get("filters") or {}
        kb_filters = KnowledgeBaseFilters(
            topic=filters.get("topic"),
            org=filters.get("org"),
            year_min=filters.get("year_min"),
            year_max=filters.get("year_max"),
        )
        results = self.kb.search(query, k=k, filters=kb_filters)
        logger.info(f"KB search completed: found {len(results)} results")
        return {"results": [r.model_dump() for r in results]}

    async def _kb_get_document(self, params: Dict) -> Dict:
        doc_id = params.get("doc_id", "")
        logger.info(f"KB get_document: doc_id='{doc_id}'")
        
        if not self.kb.is_available():
            logger.warning(f"KB get_document requested for '{doc_id}' but KB not configured")
            raise KnowledgeBaseError(
                code="KB_NOT_CONFIGURED",
                message="Knowledge base is not configured. Please set KB_VECTORSTORE_URL in your environment.",
            )
        doc = self.kb.get_document(doc_id)
        logger.info(f"KB get_document completed: doc_id='{doc_id}', title='{doc.title[:50]}...'")
        return doc.model_dump()

    async def _eval_deception(self, params: Dict) -> Dict:
        text_preview = params["text"][:50] if params.get("text") else ""
        logger.info(f"Eval deception: text='{text_preview}...'")
        result = self.evals.eval_deception(params["text"], context=params.get("context"))
        logger.info(f"Eval deception completed: score={result.get('score', 0):.2f}, label={result.get('label', 'unknown')}")
        return result

    async def _eval_dual_use(self, params: Dict) -> Dict:
        text_preview = params["text"][:50] if params.get("text") else ""
        logger.info(f"Eval dual_use: text='{text_preview}...'")
        result = self.evals.eval_dual_use(params["text"], context=params.get("context"))
        logger.info(f"Eval dual_use completed: score={result.get('score', 0):.2f}, label={result.get('label', 'unknown')}")
        return result

    async def _eval_capability(self, params: Dict) -> Dict:
        text_preview = params["text"][:50] if params.get("text") else ""
        logger.info(f"Eval capability_risk: text='{text_preview}...'")
        result = self.evals.eval_capability_risk(params["text"], context=params.get("context"))
        logger.info(f"Eval capability_risk completed: score={result.get('score', 0):.2f}, label={result.get('label', 'unknown')}")
        return result

    async def _eval_overall(self, params: Dict) -> Dict:
        text_preview = params["text"][:50] if params.get("text") else ""
        logger.info(f"Eval overall_risk: text='{text_preview}...'")
        result = self.evals.eval_overall_risk(params["text"])
        logger.info(f"Eval overall_risk completed: label={result.get('overall_label', 'unknown')}")
        return result

    async def _eval_jailbreak(self, params: Dict) -> Dict:
        text_preview = params["text"][:50] if params.get("text") else ""
        logger.info(f"Eval jailbreak: text='{text_preview}...'")
        result = self.evals.eval_jailbreak(params["text"], context=params.get("context"))
        logger.info(f"Eval jailbreak completed: score={result.get('score', 0):.2f}, label={result.get('label', 'unknown')}")
        return result

    async def _eval_prompt_injection(self, params: Dict) -> Dict:
        text_preview = params["text"][:50] if params.get("text") else ""
        logger.info(f"Eval prompt_injection: text='{text_preview}...'")
        result = self.evals.eval_prompt_injection(params["text"], context=params.get("context"))
        logger.info(f"Eval prompt_injection completed: score={result.get('score', 0):.2f}, label={result.get('label', 'unknown')}")
        return result

    async def _eval_bias(self, params: Dict) -> Dict:
        text_preview = params["text"][:50] if params.get("text") else ""
        logger.info(f"Eval bias: text='{text_preview}...'")
        result = self.evals.eval_bias(params["text"], context=params.get("context"))
        logger.info(f"Eval bias completed: score={result.get('score', 0):.2f}, label={result.get('label', 'unknown')}")
        return result

    async def _eval_toxicity(self, params: Dict) -> Dict:
        text_preview = params["text"][:50] if params.get("text") else ""
        logger.info(f"Eval toxicity: text='{text_preview}...'")
        result = self.evals.eval_toxicity(params["text"], context=params.get("context"))
        logger.info(f"Eval toxicity completed: score={result.get('score', 0):.2f}, label={result.get('label', 'unknown')}")
        return result

    async def _interp_logit_lens(self, params: Dict) -> Dict:
        model_id = params.get("model_id", "unknown")
        layer_index = params.get("layer_index", 0)
        input_preview = params.get("input_text", "")[:30]
        logger.info(f"Interp logit_lens: model={model_id}, layer={layer_index}, input='{input_preview}...'")
        result = self.interp.logit_lens(
            model_id=model_id,
            layer_index=layer_index,
            input_text=params["input_text"],
            top_k=params.get("top_k", 10),
        )
        logger.info(f"Interp logit_lens completed: {len(result.get('tokens', []))} tokens analyzed")
        return result

    async def _interp_attention(self, params: Dict) -> Dict:
        model_id = params.get("model_id", "unknown")
        layer_index = params.get("layer_index", 0)
        head_index = params.get("head_index", 0)
        input_preview = params.get("input_text", "")[:30]
        logger.info(f"Interp attention: model={model_id}, layer={layer_index}, head={head_index}, input='{input_preview}...'")
        result = self.interp.attention(
            model_id=model_id,
            layer_index=layer_index,
            head_index=head_index,
            input_text=params["input_text"],
        )
        logger.info(f"Interp attention completed: {len(result.get('tokens', []))} tokens analyzed")
        return result

    async def _interp_activation_patching(self, params: Dict) -> Dict:
        model_id = params.get("model_id", "unknown")
        layer_index = params.get("layer_index", 0)
        logger.info(f"Interp activation_patching: model={model_id}, layer={layer_index}, type={params.get('patch_type', 'residual')}")
        result = self.interp.activation_patching(
            model_id=model_id,
            input_text=params["input_text"],
            patch_text=params["patch_text"],
            layer_index=layer_index,
            patch_position=params.get("patch_position"),
            patch_type=params.get("patch_type", "residual"),
        )
        logger.info(f"Interp activation_patching completed: effect_size={result.get('effect_size', 0):.4f}")
        return result

    async def _interp_causal_tracing(self, params: Dict) -> Dict:
        model_id = params.get("model_id", "unknown")
        layers_count = len(params.get("layers_to_trace", [])) if params.get("layers_to_trace") else "all"
        logger.info(f"Interp causal_tracing: model={model_id}, layers={layers_count}")
        result = self.interp.causal_tracing(
            model_id=model_id,
            input_text=params["input_text"],
            target_token=params.get("target_token"),
            layers_to_trace=params.get("layers_to_trace"),
        )
        important_layers = len(result.get("important_layers", []))
        logger.info(f"Interp causal_tracing completed: {important_layers} important layers identified")
        return result

    async def _interp_direct_logit_attribution(self, params: Dict) -> Dict:
        model_id = params.get("model_id", "unknown")
        layer_index = params.get("layer_index", "all")
        logger.info(f"Interp direct_logit_attribution: model={model_id}, layer={layer_index}")
        result = self.interp.direct_logit_attribution(
            model_id=model_id,
            input_text=params["input_text"],
            target_token=params.get("target_token"),
            layer_index=params.get("layer_index"),
        )
        logger.info(f"Interp direct_logit_attribution completed: {len(result.get('attributions', []))} attributions computed")
        return result

    async def _interp_residual_stream(self, params: Dict) -> Dict:
        model_id = params.get("model_id", "unknown")
        layer_index = params.get("layer_index", 0)
        analysis_type = params.get("analysis_type", "statistics")
        logger.info(f"Interp residual_stream: model={model_id}, layer={layer_index}, type={analysis_type}")
        result = self.interp.residual_stream(
            model_id=model_id,
            input_text=params["input_text"],
            layer_index=layer_index,
            analysis_type=analysis_type,
        )
        logger.info(f"Interp residual_stream completed: {result.get('sequence_length', 0)} tokens analyzed")
        return result

    async def _interp_gradient_attribution(self, params: Dict) -> Dict:
        model_id = params.get("model_id", "unknown")
        method = params.get("method", "integrated_gradients")
        logger.info(f"Interp gradient_attribution: model={model_id}, method={method}")
        result = self.interp.gradient_attribution(
            model_id=model_id,
            input_text=params["input_text"],
            target_token=params.get("target_token"),
            method=method,
        )
        logger.info(f"Interp gradient_attribution completed: {len(result.get('attributions', []))} attributions computed")
        return result

    async def _health_check(self, _: Dict) -> Dict:
        logger.debug("Health check requested")
        if self.kb.is_available():
            kb_status = "stub" if self.kb.stub else "ready"
        else:
            kb_status = "not_configured"
        eval_status = "stub" if self.evals.client.use_stub else "ready"
        interp_status = "stub" if self.interp.backend.use_stub else "ready"
        result = {"status": "ok", "kb": kb_status, "evals": eval_status, "interp": interp_status}
        logger.info(f"Health check: {result}")
        return result

    async def _gov_search(self, params: Dict) -> Dict:
        query = params.get("query", "")
        logger.info(f"Gov search: query='{query[:50]}...'")
        
        if not self.governance:
            logger.warning("Gov search requested but governance not configured")
            raise KnowledgeBaseError(code="GOV_DISABLED", message="Governance corpus not configured")
        results = self.governance.search(query, k=params.get("k", 5))
        logger.info(f"Gov search completed: found {len(results)} results")
        return {"results": [r.model_dump() for r in results]}
