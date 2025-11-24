"""MCP server entry point."""
from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict

import anyio
from mcp import types
from mcp.server import stdio
from mcp.server.lowlevel.server import Server

from core.config import TransportKind, get_settings
from core.logging import get_logger, setup_logging
from safety_evals import SafetyEvalSuite
from safety_governance.retrieval import GovernanceRetriever
from safety_interp import InterpretabilitySuite
from safety_kb.retrieval import KnowledgeBaseClient

from .tool_registry import ToolRegistry

logger = get_logger(__name__)


def build_registry() -> ToolRegistry:
    settings = get_settings()
    
    logger.info("Building tool registry...")
    
    # Knowledge Base
    kb_cfg = settings.knowledge_base()
    kb_client = KnowledgeBaseClient(
        vectorstore_url=kb_cfg.vectorstore_url,
        collection=kb_cfg.collection,
        docstore_url=kb_cfg.docstore_url,
        use_stub=kb_cfg.use_stub,
    )
    if kb_client.is_available():
        logger.info(f"Knowledge Base: {'stub mode' if kb_client.stub else 'ready'} (collection: {kb_cfg.collection})")
    else:
        logger.info("Knowledge Base: not configured (KB tools will not be available)")
    
    # Evaluations
    eval_cfg = settings.evals()
    eval_suite = SafetyEvalSuite.from_config(eval_cfg)
    if eval_suite.client.use_stub:
        logger.info("Safety Evaluations: stub mode (heuristic-based, no LLM API needed)")
    else:
        logger.info(f"Safety Evaluations: ready (model: {eval_cfg.model})")
    
    # Interpretability
    interp_cfg = settings.interpretability()
    interp_suite = InterpretabilitySuite.from_config(interp_cfg)
    if interp_suite.backend.use_stub:
        logger.info("Interpretability: stub mode (no model loaded)")
    else:
        model_id = interp_suite.backend.model_id or str(interp_suite.backend.model_dir or "unknown")
        if interp_suite.backend._artifacts:
            logger.info(f"Interpretability: model loaded ({model_id})")
        else:
            logger.info(f"Interpretability: model configured ({model_id}) - will load on first tool call")
    
    # Governance
    gov_cfg = settings.governance()
    governance = GovernanceRetriever(kb_client, gov_cfg.doc_path) if gov_cfg.doc_path else None
    if governance:
        logger.info(f"Governance: ready (doc_path: {gov_cfg.doc_path})")
    else:
        logger.info("Governance: not configured (gov.search will not be available)")
    
    registry = ToolRegistry(kb=kb_client, evals=eval_suite, interp=interp_suite, governance=governance)
    logger.info(f"Tool registry built successfully with {len(registry._tools)} tools")
    return registry


async def run_server() -> None:
    settings = get_settings()
    registry = build_registry()
    server = Server(name="ai-safety-mcp-server", version="0.1.0")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        tools = await registry.list_tools()
        logger.debug(f"List tools requested: returning {len(tools)} tools")
        return tools

    @server.call_tool()
    async def _call_tool(tool_name: str, arguments: Dict[str, Any]):
        logger.info(f"Tool call: {tool_name}", extra={"tool": tool_name, "arguments_keys": list(arguments.keys())})
        try:
            result = await registry.dispatch(tool_name, arguments)
            logger.debug(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True, extra={"tool": tool_name, "error": str(e)})
            raise

    init_options = server.create_initialization_options()
    transport_cfg = settings.transport()
    
    logger.info(f"Starting MCP server (transport: {transport_cfg.kind}, version: 0.1.0)")
    
    if transport_cfg.kind is TransportKind.STDIO:
        logger.info("Server ready - waiting for MCP client connections via stdio")
        async with stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)
    else:
        logger.error(f"Unsupported transport: {transport_cfg.kind}")
        raise NotImplementedError("TCP transport not implemented yet")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Safety MCP Server")
    parser.add_argument("--log-level", default=None, help="Override log level (default from env)")
    args = parser.parse_args()

    settings = get_settings()
    log_level = args.log_level or settings.log_level
    setup_logging(log_level)
    
    logger.info("=" * 60)
    logger.info("AI Safety MCP Server starting...")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Transport: {settings.mcp_transport}")
    logger.info("=" * 60)

    try:
        anyio.run(run_server)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user, shutting down gracefully")
    except Exception as e:
        logger.critical(f"Server crashed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
