"""JSON Schemas for MCP tool IO."""
from __future__ import annotations

KB_SEARCH_INPUT = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
        "offset": {"type": "integer", "minimum": 0, "default": 0},
        "order_by": {
            "type": "string",
            "enum": ["score", "year_desc", "year_asc"],
            "default": "score",
        },
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
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "year": {"type": "integer"},
                    "org": {"type": "string"},
                    "topic": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                    "score": {"type": "number"},
                    "source": {"type": "string"},
                    "metadata": {"type": "object", "additionalProperties": True},
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
        "rationale": {"type": "string"},
        "explanation": {"type": "string"},  # Deprecated, use rationale
        "dimensions": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
        },
        "metadata": {
            "type": "object",
            "additionalProperties": True,
        },
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

# New interpretability tool schemas
ACTIVATION_PATCHING_INPUT = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "input_text": {"type": "string"},
        "patch_text": {"type": "string"},
        "layer_index": {"type": "integer", "minimum": 0},
        "patch_position": {"type": ["integer", "null"], "minimum": 0},
        "patch_type": {"type": "string", "enum": ["residual", "mlp", "attn"], "default": "residual"},
    },
    "required": ["model_id", "input_text", "patch_text", "layer_index"],
    "additionalProperties": False,
}

CAUSAL_TRACING_INPUT = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "input_text": {"type": "string"},
        "target_token": {"type": ["string", "null"]},
        "layers_to_trace": {"type": ["array", "null"], "items": {"type": "integer"}},
    },
    "required": ["model_id", "input_text"],
    "additionalProperties": False,
}

DIRECT_LOGIT_ATTRIBUTION_INPUT = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "input_text": {"type": "string"},
        "target_token": {"type": ["string", "null"]},
        "layer_index": {"type": ["integer", "null"], "minimum": 0},
    },
    "required": ["model_id", "input_text"],
    "additionalProperties": False,
}

RESIDUAL_STREAM_INPUT = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "input_text": {"type": "string"},
        "layer_index": {"type": "integer", "minimum": 0},
        "analysis_type": {
            "type": "string",
            "enum": ["statistics", "norm", "cosine_similarity"],
            "default": "statistics",
        },
    },
    "required": ["model_id", "input_text", "layer_index"],
    "additionalProperties": False,
}

GRADIENT_ATTRIBUTION_INPUT = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "input_text": {"type": "string"},
        "target_token": {"type": ["string", "null"]},
        "method": {
            "type": "string",
            "enum": ["gradient", "integrated_gradients", "saliency"],
            "default": "integrated_gradients",
        },
    },
    "required": ["model_id", "input_text"],
    "additionalProperties": False,
}
