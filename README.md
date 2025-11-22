# AI Safety MCP Server

A transport-agnostic Model Context Protocol (MCP) server exposing knowledge base, safety evaluation, interpretability, and governance tools tailored to AI Safety research assistants and agentic systems.

## Why this exists

Modern AI Safety workflows rely on quick access to research corpora, consistent safety evaluations, and lightweight interpretability probes. Instead of hand-rolling integrations per agent, this server centralizes those capabilities behind MCP tools consumable via stdio (default) or future transports.

## Feature highlights

- **Knowledge base tools (`kb.*`)** – semantic retrieval over an AI Safety corpus (default Chroma backend) plus document inspection utilities.
- **Safety evaluation tools (`eval.*`)** – deception, dual-use, capability, and aggregate risk models layered over configurable LLM classifiers with heuristic fallbacks.
- **Interpretability tools (`interp.*`)** – logit lens and attention probes for causal LMs, designed to integrate with `transformers` weights hosted locally.
- **Governance retrieval (`gov.*`)** – optional policy/document lookup reusing the knowledge base stack.
- **Health + observability** – structured logging, error codes, and a `sys.health_check` tool for orchestrators.

## Relationship to existing open-source safety tooling

This project is designed to complement, not replace, established safety toolkits:

- [OpenAI Evals](https://github.com/openai/evals) – provides a broad eval harness; AI-Safety-MCP exposes its scores via MCP-ready tools for downstream agents.
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) – the interpretability utilities borrow ideas from TransformerLens-style logit lens and attention-head probing.
- [Safety Gymnasium](https://github.com/Farama-Foundation/Safety-Gymnasium) – informs the risk taxonomies and evaluation protocols, especially for capability risk scoring.

Where practical, adapters or data pipelines can ingest outputs from these projects into the MCP knowledge base or evaluation layers.

## Repository layout

```
ai-safety-mcp-server/
└── src/
    ├── core/               # config, logging, error utilities
    ├── mcp_server/         # transport + tool registry
    ├── safety_kb/          # retrievers, doc models, ingestion helpers
    ├── safety_evals/       # deception, dual-use, capability, aggregate risk
    ├── safety_interp/      # interpretability primitives
    └── safety_governance/  # policy/governance retrieval helpers
```

## Quickstart

1. **Install dependencies**
   ```bash
   uv pip install -r <(uv pip compile pyproject.toml)
   ```
   or `pip install -e .[test]` if you prefer virtualenvs.

2. **Configure environment**
   ```bash
   cp .env.example .env
   # edit KB_VECTORSTORE_URL, LITELLM_API_KEY, etc.
   ```

3. **Launch the server (stdio)**
   ```bash
   python -m mcp_server.server
   ```

4. **Connect from an MCP-aware client** (e.g., Cursor, Co-Scientist graph) and invoke tools such as:
   ```python
   results = await client.call_tool("kb.search", {"query": "deceptive alignment"})
   risk = await client.call_tool("eval.overall_risk", {"text": hypothesis})
   ```

## Development workflow

- Run unit tests with `pytest`.
- Use `USE_*_STUB` env flags to force in-memory fixtures during local dev or CI.
- Extend the knowledge base by placing ingestion scripts under `safety_kb/indexing.py` and referencing them in docs.

## Roadmap

- TCP transport implementation behind the existing abstraction.
- Streaming logit lens visualizations and governance-specific heuristics.
- Deep integration with ARC Evals / CAIS threat models once their APIs stabilize.

## License

Apache-2.0 (see `LICENSE`).
