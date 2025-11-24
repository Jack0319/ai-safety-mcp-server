# AI Safety MCP Server

A comprehensive Model Context Protocol (MCP) server exposing knowledge base, safety evaluation, mechanistic interpretability, and governance tools tailored to AI Safety research assistants and agentic systems.

## Why this exists

Modern AI Safety workflows rely on quick access to research corpora, consistent safety evaluations, and advanced interpretability probes. Instead of hand-rolling safety and interpretability integrations for each agent, this server centralizes them behind a consistent MCP API.

## Quick Deployment

### Recommended: Docker (Simplest)

**Prerequisites**: Docker and Docker Compose installed

```bash
# 1. Clone repository
git clone https://github.com/your-org/ai-safety-mcp-server.git
cd ai-safety-mcp-server

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings (optional - works with stubs)

# 3. Start server
docker-compose up -d

# 4. Verify it's running
docker-compose logs -f
```

**Connect from MCP client** (see [MCP Client Configuration](#mcp-client-configuration) below).

### Alternative: Local Python Installation

```bash
# 1. Install
git clone https://github.com/your-org/ai-safety-mcp-server.git
cd ai-safety-mcp-server
pip install -e .

# 2. Configure (optional)
cp .env.example .env

# 3. Test
python scripts/test_all_tools.py

# 4. Run
python -m mcp_server.server
```

### MCP Client Configuration

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "ai-safety": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/ai-safety-mcp-server",
      "env": {
        "PYTHONPATH": "/path/to/ai-safety-mcp-server/src"
      }
    }
  }
}
```

**Cursor IDE** (Settings → Features → Model Context Protocol):

```json
{
  "mcpServers": {
    "ai-safety": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/ai-safety-mcp-server"
    }
  }
}
```

### Environment Variables

**Minimal configuration** (works with stubs for testing):
```bash
# No configuration needed - uses stub mode by default
```

**Full configuration** (for production):
```bash
# LLM API for safety evaluations (eval.* tools)
# LITELLM is ONLY needed for eval.* tools, NOT for interp.* tools
# Why LITELLM? It provides a unified interface to call LLM APIs (OpenAI, Anthropic, etc.)
# for safety evaluations. Interpretability tools use local models via HuggingFace.
LITELLM_API_KEY=your_api_key_here
SAFETY_EVAL_MODEL=anthropic/claude-3-opus

# Knowledge Base (optional)
KB_VECTORSTORE_URL=sqlite:///ai_safety_kb.db
KB_COLLECTION=ai_safety_docs

# Interpretability (optional)
# Can be:
# - Local path: /path/to/model or ./models/gpt2
# - HuggingFace model ID: gpt2, mistralai/Mistral-7B-v0.1, etc.
# - If not set, uses stub mode (no actual model needed)
INTERP_MODEL_DIR=gpt2  # or /path/to/local/model

# Logging
LOG_LEVEL=INFO
```

**See [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) for detailed model configuration guide.**

See [Full Deployment Guide](#full-deployment-guide) for production deployment options (VPS, systemd, cloud platforms).

## Architecture at a Glance

### Core Components

**MCP Server Core** (`mcp_server/`): The central hub that:
- Implements the Model Context Protocol (MCP) transport layer
- Maintains a `ToolRegistry` that dispatches tool calls to appropriate handlers
- Manages tool schemas, validation, and error handling

**Tool Registry Pattern**: Tools are registered via a `ToolRegistry` that takes a name, input schema, output schema, and async handler function. This makes it straightforward to add new tools.

### Optional Modules

- **Knowledge Base** (`safety_kb/`): Semantic search over AI Safety research corpus (optional)
- **Safety Evaluations** (`safety_evals/`): Risk assessment tools (deception, dual-use, capability, etc.)
- **Mechanistic Interpretability** (`safety_interp/`): Model introspection tools (logit lens, attention, activation patching, etc.)
- **Governance** (`safety_governance/`): Policy/document retrieval (optional, uses KB backend)

### Primary Use Case

This server is designed for:
- **AI Safety co-scientists**: Research assistants that help with safety analysis
- **Multi-Agent Systems (MAS)**: Agentic workflows that need safety checks and interpretability
- **Safety-focused assistants**: Tools that help researchers evaluate and understand model behavior

### Comparison: Direct LLM API vs. This Server

**Direct approach**: Call LLM API directly + use eval libraries separately
- ❌ Inconsistent interfaces across tools
- ❌ No standardized schemas
- ❌ Requires managing multiple dependencies
- ❌ Hard to compose in agent workflows

**This server**: Unified MCP interface
- ✅ Consistent tool interface via MCP
- ✅ Standardized schemas and error handling
- ✅ Single dependency for agents
- ✅ Composable tools that work together
- ✅ Local-first defaults (SQLite, local models)

### Architecture Diagram

```
[MCP Client] <-> [MCP Server Core]
                      |   |   |
                   [KB] [Eval] [Interp]
                      |
                 [Governance]
```

**Data Flow:**
1. MCP client sends tool request (e.g., `eval.deception`)
2. Server core validates input against schema
3. Tool registry dispatches to appropriate handler
4. Handler processes request (may call LLM API, load model, query KB)
5. Response is validated and returned to client

### Supported Models & Runtime Assumptions

**Model Types:**
- ✅ **Decoder-only Transformers** (GPT-2, GPT-Neo, LLaMA, Mistral, etc.)
- ✅ **Causal language models** with standard HuggingFace `transformers` interface
- ⚠️ **Encoder-decoder models**: TBD (future support)
- ❌ **Black-box API models**: Interpretability tools require local model weights

**Runtime Environment:**
- **PyTorch**: Required for interpretability tools (`interp.*`)
- **HuggingFace Transformers**: Standard model loading interface for `interp.*` tools
- **Local weights or HuggingFace**: Models can be local paths or HuggingFace model IDs for `interp.*` tools
- **LiteLLM**: Required ONLY for evaluation tools (`eval.*`) - provides unified interface to LLM APIs (OpenAI, Anthropic, etc.)

**Why LiteLLM?**
- **For `eval.*` tools**: LiteLLM provides a unified interface to call various LLM APIs (OpenAI, Anthropic, Google, etc.) for safety evaluations
- **NOT needed for `interp.*` tools**: Interpretability tools use local models loaded via HuggingFace Transformers
- **Optional**: If you don't need real LLM-based evaluations, you can use stub mode (heuristic-based) without LiteLLM

**Scale Assumptions:**
- **Single node**: Designed for single-machine deployment
- **Not distributed**: No built-in support for distributed inference or multi-node setups
- **Concurrent requests**: Handles multiple concurrent tool calls via async/await

## Design Principles

1. **Local-first and auditable**: Defaults to SQLite + local models where possible, enabling full auditability and control
2. **Composable and tool-agnostic**: Plays nicely with OpenAI Evals, TransformerLens, Captum, and other established toolkits
3. **Safety-preserving by design**: No tools that directly produce exploit code; eval tools emphasize risk detection and governance
4. **Open and extensible**: Clear plugin patterns for adding new tools, backends, and evaluation dimensions

## Feature highlights

### Knowledge Base Tools (`kb.*`) - Optional
- **Semantic search** over an AI Safety corpus using SQLite with vector embeddings (sentence-transformers) for local, dependency-free operation
- **Document retrieval** by ID with full text and metadata
- **Filtering** by topic, organization, year range
- **Pagination and ordering** support for large result sets

**Note**: KB tools are only available if `KB_VECTORSTORE_URL` is configured. The server works without KB for evaluation and interpretability tools.

### Safety Evaluation Tools (`eval.*`)
- **Deception detection** – Estimate deception or obfuscation risk
- **Dual-use assessment** – Assess misuse risk for dual-use technologies
- **Capability risk** – Score dangerous capability development risk
- **Overall risk** – Aggregate deception, dual-use, and capability risk
- **Jailbreak detection** – Detect attempts to bypass safety filters
- **Prompt injection detection** – Identify prompt injection attempts
- **Bias detection** – Detect biased or discriminatory content
- **Toxicity detection** – Identify toxic, rude, or harmful content

All evaluation tools use configurable LLM classifiers with heuristic fallbacks for offline operation.

### Mechanistic Interpretability Tools (`interp.*`)
- **Logit Lens** – Inspect intermediate layer logits for causal LMs
- **Attention Visualization** – Inspect attention weights for specific layers/heads
- **Activation Patching** – Patch activations to test causal effects (activation intervention)
- **Causal Tracing** – Trace causal effects through the model by corrupting and restoring activations
- **Direct Logit Attribution (DLA)** – Compute direct logit attribution from specific layers
- **Residual Stream Analysis** – Analyze residual stream statistics, norms, or cosine similarities
- **Gradient Attribution** – Compute gradient-based attributions (gradient, integrated gradients, saliency)

### Governance Tools (`gov.*`)
- **Policy search** – Optional policy/document lookup reusing the knowledge base stack

**Note**: `gov.search` operates over a distinct governance/policy collection (configured via `GOVERNANCE_DOC_PATH`), but shares the same vectorstore backend and schema as `kb.search`. If governance documents are not configured, `gov.search` will not be available.

### System Tools (`sys.*`)
- **Health check** – Report backend readiness for KB/Eval/Interp subsystems

## Complete Tool Reference

### Knowledge Base Tools

**Note**: KB tools are only available if a knowledge base is configured. If `KB_VECTORSTORE_URL` is not set, these tools will not be registered.

#### `kb.search`
Semantic search over the AI Safety knowledge base.

**Input:**
- `query` (string, required): Search query
- `k` (integer, optional, default: 5): Number of results (1-50)
- `offset` (integer, optional, default: 0): Pagination offset
- `order_by` (string, optional, default: "score"): Ordering ("score", "year_desc", "year_asc")
- `filters` (object, optional):
  - `topic` (string): Filter by topic
  - `org` (string): Filter by organization
  - `year_min` (integer): Minimum year
  - `year_max` (integer): Maximum year

**Output Schema:**
```json
{
  "results": [
    {
      "doc_id": "anthropic-2024-sleeper-agents",
      "title": "Sleeper Agents: Training Deceptive Models...",
      "authors": ["..."],
      "year": 2024,
      "org": "Anthropic",
      "topic": "deception",
      "url": "https://...",
      "score": 0.78,
      "snippet": "We find that...",
      "metadata": {
        "tags": ["interpretability", "evals"]
      }
    }
  ]
}
```

#### `kb.get_document`
Retrieve a document by doc_id, returning full text.

**Input:**
- `doc_id` (string, required): Document identifier

**Output Schema:**
```json
{
  "doc_id": "anthropic-2024-sleeper-agents",
  "title": "Sleeper Agents: Training Deceptive Models...",
  "authors": ["..."],
  "year": 2024,
  "org": "Anthropic",
  "topic": "deception",
  "url": "https://...",
  "text": "Full document text...",
  "metadata": {
    "tags": ["interpretability", "evals"],
    "source": "arXiv"
  }
}
```

### Safety Evaluation Tools

All evaluation tools share the same input schema:

**Input:**
- `text` (string, required): Text to evaluate
- `context` (string, optional): Additional context

**Standard Output Schema:**
```json
{
  "score": 0.82,
  "label": "high",
  "rationale": "Short natural language justification",
  "dimensions": {
    "deception": 0.9,
    "coercion": 0.7
  },
  "sources": [
    "llm_classifier:v1:claude-3-opus",
    "heuristic:keyword-blacklist:v2"
  ],
  "metadata": {
    "model_latency_ms": 532,
    "eval_version": "eval.deception:v0.3.1"
  }
}
```

**Label Thresholds:**
- **low**: 0.0 - 0.33
- **medium**: 0.33 - 0.66
- **high**: 0.66 - 1.0

Thresholds can be overridden via `EVAL_LABEL_THRESHOLDS` environment variable (format: `low_max=0.33,medium_max=0.66`).

**Fields:**
- `score` (float, 0-1): Risk score
- `label` (string): "low", "medium", or "high" (based on thresholds)
- `rationale` (string): Natural language justification (preferred over "explanation")
- `dimensions` (object, optional): Sub-dimension scores for multi-faceted risks
- `sources` (array, optional): List of evaluation methods/models used
- `metadata` (object, optional): Operational metadata (latency, versions, etc.)

#### `eval.deception`
Estimate deception or obfuscation risk for a text snippet.

#### `eval.dual_use`
Assess dual-use misuse risk for a text snippet.

**Risk Taxonomy:**
- `bio`: Biological weapons, pathogens, toxins
- `cyber`: Malware, exploits, hacking tools
- `autonomous_agents`: Autonomous systems, weaponization
- `social_engineering`: Phishing, manipulation, disinformation
- `surveillance`: Mass surveillance, privacy violations

**Output includes:**
- Standard fields plus `risk_areas` (array of strings from taxonomy above)

#### `eval.capability_risk`
Score dangerous capability development risk.

**Risk Taxonomy:**
- `autonomous_replication`: Self-replication, self-improvement
- `weapons_development`: Weapon design, manufacturing
- `bio_lab`: Biological laboratory capabilities
- `cyber_offensive`: Offensive cybersecurity capabilities
- `social_manipulation`: Large-scale social manipulation
- `escalation`: Capability escalation, recursive self-improvement

**Output includes:**
- Standard fields plus `risk_areas` (array of strings from taxonomy above)

#### `eval.overall_risk`
Aggregate deception, dual-use, and capability risk. Returns `components` object with individual scores.

#### `eval.jailbreak`
Detect jailbreak attempts designed to bypass safety filters.

#### `eval.prompt_injection`
Detect prompt injection attempts to manipulate model behavior.

#### `eval.bias`
Detect biased or discriminatory content in text.

#### `eval.toxicity`
Detect toxic, rude, or harmful content in text.

### Mechanistic Interpretability Tools

**Common Specifications:**
- **Batching**: Currently single-input only (`input_text` is a string, not a list). Batch support planned for future versions.
- **Sequence Limits**: Maximum sequence length depends on model context window (typically 2048-4096 tokens). Inputs are truncated to model max length with a warning.
- **Error Handling**: 
  - Invalid `layer_index`: Returns error with valid range (0 to num_layers-1)
  - Invalid `head_index`: Returns error with valid range (0 to num_heads-1)
  - Model not loaded: Returns error suggesting `INTERP_MODEL_DIR` configuration
- **Model Loading**: 
  - Models are loaded lazily on first tool call, then cached in memory
  - Supports both local paths and HuggingFace model IDs (e.g., `gpt2`, `mistralai/Mistral-7B-v0.1`)
  - Use `sys.health_check` to verify model availability
  - If `INTERP_MODEL_DIR` is not set, tools run in stub mode
- **Target Specification**: Tools accept either:
  - `target_token` (string): Resolved to last occurrence in sequence (with warning if multiple)
  - `target_position` (integer): Explicit token position (0-indexed, -1 for last token)

#### `interp.logit_lens`
Inspect intermediate layer logits for a causal LM.

**Input:**
- `model_id` (string, required): Model identifier (must match `INTERP_MODEL_DIR` or be "stub" for stub mode)
- `layer_index` (integer, required, min: 0): Layer to inspect
- `input_text` (string, required): Input text (single string, max length model-dependent)
- `top_k` (integer, optional, default: 10): Number of top predictions (1-50)

**Output:** Tokens and top predictions with logits and probabilities.

#### `interp.attention`
Inspect attention weights for a given layer/head.

**Input:**
- `model_id` (string, required): Model identifier
- `layer_index` (integer, required, min: 0): Layer to inspect
- `head_index` (integer, required, min: 0): Attention head index
- `input_text` (string, required): Input text

**Output:** Tokens, attention weights, and summary statistics.

#### `interp.activation_patching`
Patch activations to test causal effects (activation intervention).

**Input:**
- `model_id` (string, required): Model identifier
- `input_text` (string, required): Base text to patch into
- `patch_text` (string, required): Source text to extract activations from
- `layer_index` (integer, required, min: 0): Layer to patch at
- `patch_position` (integer, optional): Position to patch (null = last position)
- `patch_type` (string, optional, default: "residual"): Type of activation ("residual", "mlp", "attn")

**Output:** Original output, patched output, effect size, and probability distributions.

#### `interp.causal_tracing`
Trace causal effects through the model by corrupting and restoring activations.

**Input:**
- `model_id` (string, required): Model identifier
- `input_text` (string, required): Input text to trace
- `target_token` (string, optional): Token to measure effect on (resolved to last occurrence)
- `target_position` (integer, optional): Explicit token position (-1 for last token)
- `layers_to_trace` (array of integers, optional): Specific layers to trace (null = all layers)

**Output:** Causal scores, important layers, and summary.

**Note**: For long-running analyses, consider implementing async mode (future: `interp.get_result`).

#### `interp.direct_logit_attribution`
Compute direct logit attribution from specific layers (DLA).

**Input:**
- `model_id` (string, required): Model identifier
- `input_text` (string, required): Input text
- `target_token` (string, optional): Token to attribute to (resolved to last occurrence)
- `target_position` (integer, optional): Explicit token position (-1 for last token, null = top predicted)
- `layer_index` (integer, optional): Layer to compute attribution from (null = all layers)

**Output:** Attributions, top contributors, and summary.

#### `interp.residual_stream`
Analyze residual stream statistics, norms, or similarities at a specific layer.

**Input:**
- `model_id` (string, required): Model identifier
- `input_text` (string, required): Input text
- `layer_index` (integer, required, min: 0): Layer to analyze
- `analysis_type` (string, optional, default: "statistics"): Type of analysis ("statistics", "norm", "cosine_similarity")

**Output:** Statistics, norms, or cosine similarities depending on analysis type.

#### `interp.gradient_attribution`
Compute gradient-based attributions.

**Input:**
- `model_id` (string, required): Model identifier
- `input_text` (string, required): Input text
- `target_token` (string, optional): Token to attribute to (resolved to last occurrence)
- `target_position` (integer, optional): Explicit token position (-1 for last token, null = top predicted)
- `method` (string, optional, default: "integrated_gradients"): Attribution method ("gradient", "integrated_gradients", "saliency")

**Output:** Attributions, top attributed tokens, and summary.

**Performance Note**: Integrated gradients uses 50 steps by default. For faster results, use `method: "gradient"` or `method: "saliency"`.

### Governance Tools

#### `gov.search`
Search governance/policy documents (requires governance corpus configuration).

**Input:** Same as `kb.search`

**Output:** Same as `kb.search`

### System Tools

#### `sys.health_check`
Report backend readiness for KB/Eval/Interp subsystems.

**Input:** None

**Output:**
- `status` (string): Overall status ("ok")
- `kb` (string): Knowledge base status ("ready" or "stub")
- `evals` (string): Evaluation status ("ready" or "stub")
- `interp` (string): Interpretability status ("ready" or "stub")

## Relationship to existing open-source safety tooling

This project is designed to complement, not replace, established safety toolkits:

- [OpenAI Evals](https://github.com/openai/evals) – provides a broad eval harness; AI-Safety-MCP exposes its scores via MCP-ready tools for downstream agents.
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) – the interpretability utilities borrow ideas from TransformerLens-style logit lens and attention-head probing.
- [Safety Gymnasium](https://github.com/Farama-Foundation/Safety-Gymnasium) – informs the risk taxonomies and evaluation protocols, especially for capability risk scoring.

Where practical, adapters or data pipelines can ingest outputs from these projects into the MCP knowledge base or evaluation layers.

## Research References

### Implementation Notes

This implementation provides MCP-accessible wrappers around established interpretability methodologies. For production research, we recommend:

1. **Comparing with reference implementations**: Our tools follow the methodologies from the cited papers, but for rigorous research, compare results with:
   - [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for activation patching, logit lens, and attention analysis
   - [Captum](https://github.com/pytorch/captum) for gradient attribution methods
   - Anthropic's published code and notebooks for causal tracing

2. **Using established libraries**: For critical research, consider using the reference implementations directly alongside this MCP server for validation.

3. **Methodology alignment**: Our implementations aim to match the cited methodologies, but may have simplifications for MCP integration. Always verify results match expected behavior from reference implementations.

### Mechanistic Interpretability

Our interpretability tools are based on established research and methodologies:

**Logit Lens:**
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
- TransformerLens implementation: https://github.com/neelnanda-io/TransformerLens

**Attention Analysis:**
- Vaswani, A., et al. (2017). "Attention is All You Need." *NeurIPS*.
- Clark, K., et al. (2019). "What Does BERT Look At?" *ACL*.

**Activation Patching:**
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." https://transformer-circuits.pub/2021/framework/index.html
- Olsson, C., et al. (2022). "In-context Learning and Induction Heads." https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Anthropic's activation patching methodology

**Causal Tracing:**
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." https://transformer-circuits.pub/2021/framework/index.html
- Meng, K., et al. (2022). "Locating and Editing Factual Associations in GPT." *arXiv:2202.05262*
- Anthropic's causal tracing methodology

**Direct Logit Attribution:**
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." https://transformer-circuits.pub/2021/framework/index.html
- Anthropic's interpretability research

**Gradient Attribution:**
- Sundararajan, M., et al. (2017). "Axiomatic Attribution for Deep Networks." *ICML* (Integrated Gradients)
- Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." *ICCV*
- Adebayo, J., et al. (2018). "Sanity Checks for Saliency Maps." *NeurIPS*
- Captum library: https://github.com/pytorch/captum

**Residual Stream:**
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." https://transformer-circuits.pub/2021/framework/index.html
- Anthropic's mechanistic interpretability research

### Safety Evaluation

Our evaluation methodologies draw from:
- Anthropic's safety research and red teaming practices
- OpenAI's safety guidelines and evaluation frameworks
- Academic research on AI safety, alignment, and risk assessment

## Repository layout

```
ai-safety-mcp-server/
└── src/
    ├── core/               # config, logging, error utilities
    ├── mcp_server/         # transport + tool registry
    ├── safety_kb/          # retrievers, doc models, ingestion helpers
    ├── safety_evals/       # deception, dual-use, capability, jailbreak, etc.
    ├── safety_interp/      # interpretability primitives (logit lens, attention, patching, etc.)
    └── safety_governance/  # policy/governance retrieval helpers
```

## Full Deployment Guide

This section covers production deployment options for self-hosting.

### Why Self-Hosting?

Self-hosting provides:
- **Full control** over your data, models, and infrastructure
- **Data privacy** - all data stays on your infrastructure
- **Customization** - modify and extend the server to your needs
- **No vendor lock-in** - use any hosting provider
- **Cost control** - pay only for what you use

### Hosting Requirements

**Minimum Requirements:**
- **CPU**: 2 vCPUs (4+ recommended for interpretability)
- **RAM**: 2 GB (4-8 GB recommended for interpretability with models)
- **Storage**: 20 GB (40+ GB if storing models locally)
- **Network**: Stable internet connection for LLM API calls

**Budget Guidance** (order-of-magnitude estimates):
- Basic (evaluations only): ~$10-20/month
- With KB: ~$15-30/month
- With interpretability: ~$30-50/month (GPU instances cost significantly more)

### Production Deployment Options

#### Option 1: Cloud VPS (Recommended for Full Features)

**Providers**: DigitalOcean, Linode, Hetzner, Vultr, AWS EC2, Google Cloud Compute

**Quick Setup:**
   ```bash
# On your VPS (Ubuntu 22.04+)
git clone https://github.com/your-org/ai-safety-mcp-server.git
cd ai-safety-mcp-server
python3 -m venv venv && source venv/bin/activate
   pip install -e .
cp .env.example .env && nano .env  # Configure

# Set up systemd service
sudo tee /etc/systemd/system/ai-safety-mcp.service > /dev/null <<EOF
[Unit]
Description=AI Safety MCP Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python -m mcp_server.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ai-safety-mcp
sudo systemctl start ai-safety-mcp
```

#### Option 2: Docker (Already covered in Quick Deployment)

See [Quick Deployment](#quick-deployment) section above.

#### Option 3: Container Hosting (Railway, Render, Fly.io)

**Railway:**
```bash
npm i -g @railway/cli
railway login && railway init
railway variables set LITELLM_API_KEY=your_key
railway up
```

**Render**: Connect GitHub repo, set environment variables, deploy.

**Fly.io:**
   ```bash
fly launch
fly secrets set LITELLM_API_KEY=your_key
fly deploy
```

### Logging & Observability

The server provides comprehensive structured logging (JSON format) for monitoring and debugging.

**Log Levels:**
- `DEBUG`: Detailed diagnostic information (tool dispatches, model ID checks, etc.)
- `INFO`: General operational information (server startup, tool calls, model loading)
- `WARNING`: Warning conditions (missing configurations, model ID mismatches)
- `ERROR`: Error conditions (tool failures, model loading errors)
- `CRITICAL`: Critical failures (server crashes)

**What Gets Logged:**

**Server Startup:**
- Configuration status for KB, Evaluations, Interpretability, Governance
- Number of tools registered
- Transport mode
- Model configuration (if interpretability is enabled)

**Tool Calls:**
- Every tool call with tool name and argument keys
- Tool completion status
- Evaluation results (scores, labels)
- KB search queries and result counts
- Interpretability operations (model, layer, analysis type)

**Model Operations:**
- Model loading (lazy loading on first use)
- Loading time and source (local vs HuggingFace)
- Model loading errors with helpful messages

**Errors:**
- Tool failures with full stack traces
- Configuration errors
- Model loading failures
- API call failures (for evaluations)

**Example Log Output:**
```json
{"timestamp":"2025-11-24T23:00:25-0600","level":"INFO","logger":"mcp_server.server","message":"Building tool registry..."}
{"timestamp":"2025-11-24T23:00:25-0600","level":"INFO","logger":"mcp_server.server","message":"Knowledge Base: ready (collection: ai_safety_docs)"}
{"timestamp":"2025-11-24T23:00:25-0600","level":"INFO","logger":"mcp_server.server","message":"Tool call: eval.deception","tool":"eval.deception","arguments_keys":["text"]}
{"timestamp":"2025-11-24T23:00:25-0600","level":"INFO","logger":"mcp_server.tool_registry","message":"Eval deception completed: score=0.25, label=low"}
```

**Log Configuration:**
- Set `LOG_LEVEL` environment variable (DEBUG, INFO, WARNING, ERROR)
- Or use `--log-level` command-line argument
- Logs are written to stderr in JSON format for easy parsing

### Monitoring & Maintenance

**Health Check:**
   ```bash
# Check service status
systemctl status ai-safety-mcp  # VPS
docker-compose ps                # Docker

# View logs
journalctl -u ai-safety-mcp -f   # VPS
docker-compose logs -f           # Docker
```

**Backups:**
   ```bash
# Backup knowledge base
cp ai_safety_kb.db backups/ai_safety_kb_$(date +%Y%m%d).db
```

See [Development workflow](#development-workflow) for more details.

## Threat Model & Security Considerations

### Threat Model

This server will often process sensitive prompts from agents, including:
- User queries that may contain personal information
- Research prompts exploring safety risks
- Potentially harmful content being evaluated

**Possible Misuse:**
- Using interpretability tools to optimize jailbreaks or discover capabilities
- Extracting sensitive information from evaluation logs
- Bypassing safety checks through tool manipulation

### Mitigations

**Access Control:**
- **Recommended**: Deploy MCP server behind an authenticated proxy (mTLS, API key)
- **Future**: Environment variables `AUTH_MODE` and `AUTH_TOKEN` for built-in authentication
- Use IP allowlists for production deployments
- Restrict network access to MCP server

**Logging & Privacy:**
- **Text Redaction**: Set `LOG_TEXT_CONTENT=false` to avoid logging full text content
- **Hashing**: Optional text hashing for audit logs (future feature)
- **Truncation**: Long inputs are truncated in logs (configurable via `LOG_MAX_LENGTH`)

**Policy Gates:**
- **Optional**: Restrict interpretability tools based on evaluation scores
  - Example: `interp.*` tools only accessible if `eval.overall_risk < 0.5` for associated text
  - Future: Configurable policy engine via `POLICY_GATE_ENABLED`

**Best Practices:**
1. Never expose the MCP server directly to the internet
2. Use VPN or private networks for sensitive deployments
3. Rotate API keys regularly
4. Monitor tool usage patterns for anomalies
5. Implement rate limiting for production use


## Developer Experience & Extensibility

### Tool Registration Pattern

Tools are registered via the `ToolRegistry` class in `mcp_server/tool_registry.py`. Each tool requires:
- **Name**: Tool identifier (e.g., `eval.custom_metric`)
- **Description**: Human-readable description
- **Input Schema**: JSON Schema for validation
- **Output Schema**: JSON Schema for response (optional)
- **Handler**: Async function that processes the request

**Example: Adding a New Evaluation Tool**

   ```python
# In safety_evals/custom_metric.py
def evaluate(text: str, *, client: EvalModelClient, context: str | None = None) -> Dict[str, object]:
    score_obj = client.score_text(CUSTOM_RUBRIC, text, context=context, category="custom")
    return {
        "score": score_obj.score,
        "label": score_obj.label,
        "rationale": score_obj.explanation,
    }

# In mcp_server/tool_registry.py
from safety_evals import custom_metric

# In _register_tools():
self._add_tool(
    ToolHandler(
        name="eval.custom_metric",
        description="Evaluate custom metric for text.",
        input_schema=schemas.EVAL_INPUT,
        output_schema=schemas.EVAL_OUTPUT,
        handler=self._eval_custom_metric,
    )
)

# Add handler method:
async def _eval_custom_metric(self, params: Dict) -> Dict:
    return self.evals.eval_custom_metric(params["text"], context=params.get("context"))
```

### Plugin / Extension Patterns

**Adding a New Evaluation Tool:**
1. Create `safety_evals/your_metric.py` with `evaluate()` function
2. Add to `SafetyEvalSuite` in `safety_evals/__init__.py`
3. Register tool in `ToolRegistry._register_tools()`
4. Add schema to `mcp_server/schemas.py` if needed

**Adding a New KB Backend:**
1. Implement backend class in `safety_kb/retrieval.py` (see `ChromaBackend`, `SQLiteVectorStore`)
2. Update `KnowledgeBaseClient.__init__()` to detect and use new backend
3. Add configuration options to `core/config.py`

**Adding a New Interpretability Primitive:**
1. Create tool class in `safety_interp/your_tool.py`
2. Add to `InterpretabilitySuite` in `safety_interp/__init__.py`
3. Register tool in `ToolRegistry._register_tools()`
4. Add schema to `mcp_server/schemas.py`

**Example: Adding Sparse Autoencoder (SAE) Integration**

```python
# safety_interp/sae.py
class SAETool:
    def __init__(self, backend: InterpretabilityBackend, sae_path: Path):
        self.backend = backend
        self.sae = load_sae(sae_path)  # Your SAE loading logic
    
    def run(self, *, input_text: str, layer_index: int) -> Dict[str, object]:
        # Your SAE analysis logic
        pass

# Add to InterpretabilitySuite and register in ToolRegistry
```

### Configuration Discovery

Configuration is loaded in this order (later overrides earlier):
1. Default values in `core/config.py`
2. `config/config.yaml` (if exists, future feature)
3. Environment variables (`.env` file or system env)
4. Command-line arguments (future feature)

**Current**: Only environment variables are supported. YAML config support planned.

### Versioning & Migrations

**Server Version**: Tracked in `pyproject.toml` (`version = "0.1.0"`)

**KB Schema Migrations**: 
- Current: Manual migration scripts (see `scripts/populate_kb.py`)
- Future: Alembic-based migrations for schema changes
- Recommendation: Tag releases and pin versions in deployment

**Breaking Changes**: Documented in CHANGELOG.md (to be created)

## Development workflow

### Testing

**Comprehensive Tool Test:**
```bash
# Test all tools with a single command
python scripts/test_all_tools.py
```

This script verifies:
- All tools are properly registered
- KB tools work (if KB is configured)
- All evaluation tools function correctly
- All interpretability tools function correctly
- System health check works

**Unit Tests:**
```bash
# Run pytest unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

**Using Stubs:**
```bash
# Use stub mode for testing without external dependencies
export USE_KB_STUB=true
export USE_EVAL_STUB=true
export USE_INTERP_STUB=true
python scripts/test_all_tools.py
```

### Development Tips

- Use `USE_*_STUB` env flags to force in-memory fixtures during local dev or CI
- Extend the knowledge base by placing ingestion scripts under `safety_kb/indexing.py`
- All tools support stub mode for offline development
- Check `sys.health_check` to verify backend readiness

## Example Use Cases

### 1. Safety Evaluation Pipeline
```python
# Evaluate a user prompt for multiple safety risks
text = "How to create a computer virus?"

results = {
    "jailbreak": await client.call_tool("eval.jailbreak", {"text": text}),
    "prompt_injection": await client.call_tool("eval.prompt_injection", {"text": text}),
    "toxicity": await client.call_tool("eval.toxicity", {"text": text}),
    "overall": await client.call_tool("eval.overall_risk", {"text": text})
}
```

### 2. Mechanistic Interpretability Research
```python
# Trace causal effects through the model
trace_result = await client.call_tool("interp.causal_tracing", {
    "model_id": "gpt2",
    "input_text": "The quick brown fox jumps over the lazy dog",
    "layers_to_trace": [0, 5, 10]
})

# Analyze residual stream
residual = await client.call_tool("interp.residual_stream", {
    "model_id": "gpt2",
    "input_text": "The quick brown fox",
    "layer_index": 5,
    "analysis_type": "cosine_similarity"
})
```

### 3. Knowledge Base Research
```python
# Search for relevant safety research
papers = await client.call_tool("kb.search", {
    "query": "mechanistic interpretability",
    "k": 10,
    "filters": {
        "year_min": 2020,
        "topic": "interpretability"
    }
})

# Get full document
doc = await client.call_tool("kb.get_document", {
    "doc_id": papers["results"][0]["doc_id"]
})
```

## Roadmap

- **[MCP]** TCP transport implementation behind the existing abstraction
- **[Interp]** Streaming logit lens visualizations
- **[Interp]** Sparse Autoencoder (SAE) integration for feature visualization
- **[Interp]** Circuit discovery tools
- **[DX]** Model comparison utilities
- **[Safety]** Deep integration with ARC Evals / CAIS threat models once their APIs stabilize
- **[DX]** Web UI for tool exploration and visualization
- **[Scale]** Batch processing support for evaluations
- **[DX]** Configuration file support (YAML) in addition to environment variables
- **[Safety]** Policy gate engine for conditional tool access
- **[Interp]** Async job queue for long-running interpretability analyses
- **[MCP]** Authentication and authorization built-in support

**Legend:**
- **[MCP]**: MCP protocol features
- **[Interp]**: Interpretability tools
- **[Safety]**: Safety evaluation features
- **[DX]**: Developer experience
- **[Scale]**: Scalability and performance

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style (ruff, mypy)
- Tests are added for new features
- Documentation is updated

## License

Apache-2.0 (see `LICENSE`).
