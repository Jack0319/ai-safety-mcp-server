#!/usr/bin/env python3
"""Comprehensive test script to verify all MCP server tools work correctly."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import EvalConfig, InterpretabilityConfig, get_settings
from mcp_server.tool_registry import ToolRegistry
from safety_evals import SafetyEvalSuite
from safety_interp import InterpretabilitySuite
from safety_kb.models import Document
from safety_kb.retrieval import KnowledgeBaseClient


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_test(name: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing: {name}{Colors.RESET}")


def print_success(message: str):
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


async def test_kb_tools(registry: ToolRegistry):
    """Test knowledge base tools."""
    print_test("Knowledge Base Tools")
    
    # Check if KB is available
    if not registry.kb.is_available():
        print_warning("KB not configured - skipping KB tests")
        return True
    
    try:
        # Test kb.search
        result = await registry.dispatch("kb.search", {"query": "alignment", "k": 3})
        assert "results" in result, "kb.search should return results"
        assert isinstance(result["results"], list), "results should be a list"
        print_success("kb.search works")
        
        if result["results"]:
            # Test kb.get_document
            doc_id = result["results"][0]["doc_id"]
            doc_result = await registry.dispatch("kb.get_document", {"doc_id": doc_id})
            assert "doc_id" in doc_result, "kb.get_document should return doc_id"
            assert "text" in doc_result, "kb.get_document should return text"
            print_success("kb.get_document works")
        else:
            print_warning("No documents in KB to test kb.get_document")
        
        return True
    except Exception as e:
        print_error(f"KB tools failed: {e}")
        return False


async def test_eval_tools(registry: ToolRegistry):
    """Test safety evaluation tools."""
    print_test("Safety Evaluation Tools")
    
    test_text = "This is a test prompt that should be evaluated for safety risks."
    
    eval_tools = [
        "eval.deception",
        "eval.dual_use",
        "eval.capability_risk",
        "eval.jailbreak",
        "eval.prompt_injection",
        "eval.bias",
        "eval.toxicity",
    ]
    
    all_passed = True
    
    for tool_name in eval_tools:
        try:
            result = await registry.dispatch(tool_name, {"text": test_text})
            assert "score" in result, f"{tool_name} should return score"
            assert "label" in result, f"{tool_name} should return label"
            assert result["label"] in ["low", "medium", "high"], f"{tool_name} label should be low/medium/high"
            assert 0 <= result["score"] <= 1, f"{tool_name} score should be 0-1"
            print_success(f"{tool_name} works (score: {result['score']:.2f}, label: {result['label']})")
        except Exception as e:
            print_error(f"{tool_name} failed: {e}")
            all_passed = False
    
    # Test overall_risk
    try:
        result = await registry.dispatch("eval.overall_risk", {"text": test_text})
        assert "overall_label" in result, "eval.overall_risk should return overall_label"
        assert "components" in result, "eval.overall_risk should return components"
        assert "deception" in result["components"], "components should include deception"
        assert "dual_use" in result["components"], "components should include dual_use"
        assert "capability" in result["components"], "components should include capability"
        print_success("eval.overall_risk works")
    except Exception as e:
        print_error(f"eval.overall_risk failed: {e}")
        all_passed = False
    
    return all_passed


async def test_interp_tools(registry: ToolRegistry):
    """Test interpretability tools."""
    print_test("Interpretability Tools")
    
    test_text = "The capital of France is"
    model_id = "stub"  # Will use stub mode
    
    all_passed = True
    
    # Test logit_lens
    try:
        result = await registry.dispatch("interp.logit_lens", {
            "model_id": model_id,
            "layer_index": 0,
            "input_text": test_text,
            "top_k": 5,
        })
        assert "tokens" in result, "logit_lens should return tokens"
        assert "top_predictions" in result, "logit_lens should return top_predictions"
        print_success("interp.logit_lens works")
    except Exception as e:
        print_error(f"interp.logit_lens failed: {e}")
        all_passed = False
    
    # Test attention
    try:
        result = await registry.dispatch("interp.attention", {
            "model_id": model_id,
            "layer_index": 0,
            "head_index": 0,
            "input_text": test_text,
        })
        assert "tokens" in result, "attention should return tokens"
        assert "weights" in result, "attention should return weights"
        print_success("interp.attention works")
    except Exception as e:
        print_error(f"interp.attention failed: {e}")
        all_passed = False
    
    # Test activation_patching
    try:
        result = await registry.dispatch("interp.activation_patching", {
            "model_id": model_id,
            "input_text": test_text,
            "patch_text": "The capital of Germany is",
            "layer_index": 0,
        })
        assert "original_output" in result, "activation_patching should return original_output"
        assert "patched_output" in result, "activation_patching should return patched_output"
        assert "effect_size" in result, "activation_patching should return effect_size"
        print_success("interp.activation_patching works")
    except Exception as e:
        print_error(f"interp.activation_patching failed: {e}")
        all_passed = False
    
    # Test causal_tracing
    try:
        result = await registry.dispatch("interp.causal_tracing", {
            "model_id": model_id,
            "input_text": test_text,
        })
        assert "tokens" in result, "causal_tracing should return tokens"
        assert "causal_scores" in result, "causal_tracing should return causal_scores"
        print_success("interp.causal_tracing works")
    except Exception as e:
        print_error(f"interp.causal_tracing failed: {e}")
        all_passed = False
    
    # Test direct_logit_attribution
    try:
        result = await registry.dispatch("interp.direct_logit_attribution", {
            "model_id": model_id,
            "input_text": test_text,
        })
        assert "tokens" in result, "direct_logit_attribution should return tokens"
        assert "attributions" in result, "direct_logit_attribution should return attributions"
        print_success("interp.direct_logit_attribution works")
    except Exception as e:
        print_error(f"interp.direct_logit_attribution failed: {e}")
        all_passed = False
    
    # Test residual_stream
    try:
        result = await registry.dispatch("interp.residual_stream", {
            "model_id": model_id,
            "input_text": test_text,
            "layer_index": 0,
            "analysis_type": "statistics",
        })
        assert "tokens" in result, "residual_stream should return tokens"
        assert "statistics" in result, "residual_stream should return statistics"
        print_success("interp.residual_stream works")
    except Exception as e:
        print_error(f"interp.residual_stream failed: {e}")
        all_passed = False
    
    # Test gradient_attribution
    try:
        result = await registry.dispatch("interp.gradient_attribution", {
            "model_id": model_id,
            "input_text": test_text,
            "method": "gradient",
        })
        assert "tokens" in result, "gradient_attribution should return tokens"
        assert "attributions" in result, "gradient_attribution should return attributions"
        print_success("interp.gradient_attribution works")
    except Exception as e:
        print_error(f"interp.gradient_attribution failed: {e}")
        all_passed = False
    
    return all_passed


async def test_system_tools(registry: ToolRegistry):
    """Test system tools."""
    print_test("System Tools")
    
    try:
        result = await registry.dispatch("sys.health_check", {})
        assert "status" in result, "health_check should return status"
        assert "kb" in result, "health_check should return kb status"
        assert "evals" in result, "health_check should return evals status"
        assert "interp" in result, "health_check should return interp status"
        print_success(f"sys.health_check works (KB: {result['kb']}, Evals: {result['evals']}, Interp: {result['interp']})")
        return True
    except Exception as e:
        print_error(f"sys.health_check failed: {e}")
        return False


async def test_tool_listing(registry: ToolRegistry):
    """Test that all expected tools are listed."""
    print_test("Tool Listing")
    
    try:
        tools = await registry.list_tools()
        tool_names = [tool.name for tool in tools]
        
        expected_tools = [
            "sys.health_check",
            "eval.deception",
            "eval.dual_use",
            "eval.capability_risk",
            "eval.overall_risk",
            "eval.jailbreak",
            "eval.prompt_injection",
            "eval.bias",
            "eval.toxicity",
            "interp.logit_lens",
            "interp.attention",
            "interp.activation_patching",
            "interp.causal_tracing",
            "interp.direct_logit_attribution",
            "interp.residual_stream",
            "interp.gradient_attribution",
        ]
        
        # KB tools are optional
        if registry.kb.is_available():
            expected_tools.extend(["kb.search", "kb.get_document"])
        
        missing_tools = [t for t in expected_tools if t not in tool_names]
        if missing_tools:
            print_error(f"Missing tools: {missing_tools}")
            return False
        
        print_success(f"All {len(tool_names)} expected tools are registered")
        print(f"  Available tools: {', '.join(sorted(tool_names))}")
        return True
    except Exception as e:
        print_error(f"Tool listing failed: {e}")
        return False


async def main():
    """Run all tests."""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("AI Safety MCP Server - Comprehensive Tool Test")
    print("=" * 60)
    print(f"{Colors.RESET}")
    
    # Set up stubs for testing
    os.environ.setdefault("USE_KB_STUB", "true")
    os.environ.setdefault("USE_EVAL_STUB", "true")
    os.environ.setdefault("USE_INTERP_STUB", "true")
    
    # Create KB with stub data
    kb = KnowledgeBaseClient(
        vectorstore_url=None,
        collection="test",
        docstore_url=None,
        use_stub=True,
    )
    kb.load_stub_documents([
        Document(
            doc_id="test-doc-1",
            title="Test Document",
            text="This is a test document about AI safety and alignment.",
            metadata={"topic": "alignment", "year": 2024},
        ),
    ])
    
    # Create eval suite
    eval_suite = SafetyEvalSuite.from_config(
        EvalConfig(model="stub", litellm_api_key=None, use_stub=True)
    )
    
    # Create interp suite
    interp_suite = InterpretabilitySuite.from_config(
        InterpretabilityConfig(model_dir=None, use_stub=True)
    )
    
    # Create registry
    registry = ToolRegistry(kb=kb, evals=eval_suite, interp=interp_suite, governance=None)
    
    # Run tests
    results = {}
    
    results["tool_listing"] = await test_tool_listing(registry)
    results["system"] = await test_system_tools(registry)
    results["kb"] = await test_kb_tools(registry)
    results["eval"] = await test_eval_tools(registry)
    results["interp"] = await test_interp_tools(registry)
    
    # Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"{Colors.RESET}")
    
    all_passed = True
    for category, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {category:20s}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All tests passed!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some tests failed{Colors.RESET}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

