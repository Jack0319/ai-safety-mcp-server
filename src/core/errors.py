"""Shared error types and helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class MCPSafetyError(Exception):
    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_response(self) -> dict[str, Any]:
        payload = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class KnowledgeBaseError(MCPSafetyError):
    pass


class KnowledgeBaseNotFound(KnowledgeBaseError):
    def __init__(self, doc_id: str):
        super().__init__(
            code="KB_NOT_FOUND",
            message=f"Document with id '{doc_id}' not found.",
            details={"doc_id": doc_id},
        )


class KnowledgeBaseUnavailable(KnowledgeBaseError):
    def __init__(self, reason: str):
        super().__init__(code="KB_BACKEND_UNAVAILABLE", message=reason)


class EvaluationError(MCPSafetyError):
    pass


class EvaluationBackendUnavailable(EvaluationError):
    def __init__(self, reason: str):
        super().__init__(code="EVAL_BACKEND_UNAVAILABLE", message=reason)


class InvalidInputError(MCPSafetyError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(code="INVALID_INPUT", message=message, details=details)


class InternalServerError(MCPSafetyError):
    def __init__(self, message: str = "Internal server error"):
        super().__init__(code="INTERNAL_ERROR", message=message)


class UnsupportedModelError(MCPSafetyError):
    def __init__(self, model_id: str):
        super().__init__(
            code="UNSUPPORTED_MODEL",
            message=f"Model '{model_id}' is not supported for this tool.",
            details={"model_id": model_id},
        )


def wrap_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to coerce arbitrary exceptions into `InternalServerError`."""

    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except MCPSafetyError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise InternalServerError(str(exc)) from exc

    return wrapper
