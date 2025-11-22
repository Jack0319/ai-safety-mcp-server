"""JSON Schemas for MCP tool IO."""
from __future__ import annotations

KB_SEARCH_INPUT = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
        "filters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "org": {"type": "string"},
                "year_min": {"type": "integer"},
                "year_max": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

KB_SEARCH_OUTPUT = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                    "score": {"type": "number"},
                    "source": {"type": "string"},
                    "topic": {"type": "string"},
                },
                "required": ["doc_id", "title", "snippet", "score"],
                "additionalProperties": True,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

KB_GET_DOCUMENT_INPUT = {
    "type": "object",
    "properties": {"doc_id": {"type": "string"}},
    "required": ["doc_id"],
    "additionalProperties": False,
}

KB_GET_DOCUMENT_OUTPUT = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string"},
        "title": {"type": "string"},
        "url": {"type": "string"},
        "text": {"type": "string"},
        "metadata": {"type": "object"},
    },
    "required": ["doc_id", "title", "text"],
    "additionalProperties": True,
}

EVAL_INPUT = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "context": {"type": "string"},
    },
    "required": ["text"],
    "additionalProperties": False,
}

EVAL_OUTPUT = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string", "enum": ["low", "medium", "high"]},
        "explanation": {"type": "string"},
    },
    "required": ["score", "label"],
    "additionalProperties": False,
}

DUAL_USE_OUTPUT = {
    **EVAL_OUTPUT,
    "properties": {**EVAL_OUTPUT["properties"], "risk_areas": {"type": "array", "items": {"type": "string"}}},
}

CAPABILITY_OUTPUT = EVAL_OUTPUT

OVERALL_INPUT = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}

OVERALL_OUTPUT = {
    "type": "object",
    "properties": {
        "overall_label": {"type": "string", "enum": ["low", "medium", "high"]},
        "components": {
            "type": "object",
            "properties": {
                "deception": {"type": "number", "minimum": 0, "maximum": 1},
                "dual_use": {"type": "number", "minimum": 0, "maximum": 1},
                "capability": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "additionalProperties": False,
        },
        "explanation": {"type": "string"},
    },
    "required": ["overall_label", "components"],
    "additionalProperties": False,
}

LOGIT_LENS_INPUT = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "layer_index": {"type": "integer", "minimum": 0},
        "input_text": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
    },
    "required": ["model_id", "layer_index", "input_text"],
    "additionalProperties": False,
}

ATTENTION_INPUT = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "layer_index": {"type": "integer", "minimum": 0},
        "head_index": {"type": "integer", "minimum": 0},
        "input_text": {"type": "string"},
    },
    "required": ["model_id", "layer_index", "head_index", "input_text"],
    "additionalProperties": False,
}

HEALTH_CHECK_OUTPUT = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "kb": {"type": "string"},
        "evals": {"type": "string"},
        "interp": {"type": "string"},
    },
    "required": ["status"],
    "additionalProperties": True,
}
