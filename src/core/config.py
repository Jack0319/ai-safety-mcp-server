"""Configuration management for the AI Safety MCP server."""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TransportKind(str, Enum):
    STDIO = "stdio"
    TCP = "tcp"


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO", description="Root log level")


class EvalConfig(BaseModel):
    model: str = Field(default="anthropic/claude-3-opus")
    litellm_api_key: Optional[str] = None
    use_stub: bool = False


class KnowledgeBaseConfig(BaseModel):
    vectorstore_url: Optional[str] = None
    collection: str = Field(default="ai_safety_docs")
    docstore_url: Optional[str] = None
    use_stub: bool = False


class InterpretabilityConfig(BaseModel):
    model_dir: Optional[Path] = None
    use_stub: bool = False


class GovernanceConfig(BaseModel):
    doc_path: Optional[Path] = None


class TransportConfig(BaseModel):
    kind: TransportKind = TransportKind.STDIO
    port: Optional[int] = None


class AppSettings(BaseSettings):
    """Strongly-typed settings loaded from environment variables / .env."""

    kb_vectorstore_url: Optional[str] = Field(default=None, alias="KB_VECTORSTORE_URL")
    kb_collection: str = Field(default="ai_safety_docs", alias="KB_COLLECTION")
    kb_docstore_url: Optional[str] = Field(default=None, alias="KB_DOCSTORE_URL")

    safety_eval_model: str = Field(default="anthropic/claude-3-opus", alias="SAFETY_EVAL_MODEL")
    litellm_api_key: Optional[str] = Field(default=None, alias="LITELLM_API_KEY")

    interp_model_dir: Optional[Path] = Field(default=None, alias="INTERP_MODEL_DIR")
    governance_doc_path: Optional[Path] = Field(default=None, alias="GOVERNANCE_DOC_PATH")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    mcp_transport: Literal["stdio", "tcp"] = Field(default="stdio", alias="MCP_TRANSPORT")
    mcp_port: Optional[int] = Field(default=None, alias="MCP_PORT")

    use_kb_stub: bool = Field(default=False, alias="USE_KB_STUB")
    use_eval_stub: bool = Field(default=False, alias="USE_EVAL_STUB")
    use_interp_stub: bool = Field(default=False, alias="USE_INTERP_STUB")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    def knowledge_base(self) -> KnowledgeBaseConfig:
        return KnowledgeBaseConfig(
            vectorstore_url=self.kb_vectorstore_url,
            collection=self.kb_collection,
            docstore_url=self.kb_docstore_url,
            use_stub=self.use_kb_stub,
        )

    def evals(self) -> EvalConfig:
        return EvalConfig(
            model=self.safety_eval_model,
            litellm_api_key=self.litellm_api_key,
            use_stub=self.use_eval_stub,
        )

    def interpretability(self) -> InterpretabilityConfig:
        return InterpretabilityConfig(model_dir=self.interp_model_dir, use_stub=self.use_interp_stub)

    def governance(self) -> GovernanceConfig:
        return GovernanceConfig(doc_path=self.governance_doc_path)

    def logging(self) -> LoggingConfig:
        return LoggingConfig(level=self.log_level)

    def transport(self) -> TransportConfig:
        return TransportConfig(kind=TransportKind(self.mcp_transport), port=self.mcp_port)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached settings object so downstream modules share configuration."""

    return AppSettings()  # type: ignore[arg-type]
