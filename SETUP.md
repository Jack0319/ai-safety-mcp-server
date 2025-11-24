# AI Safety MCP Server - Setup Guide

This guide explains how to set up and use the AI Safety MCP Server with a local SQLite knowledge base.

## Architecture Overview

The server uses a **SQLite database with vector embeddings** for semantic search, eliminating the need for external vector stores like ChromaDB. This makes setup simpler and the server fully self-contained.

### Components:

1. **SQLite Vector Store**: Stores documents and their embeddings in a single `.db` file
2. **Sentence Transformers**: Generates embeddings using the `all-MiniLM-L6-v2` model
3. **MCP Server**: Exposes tools via the Model Context Protocol (stdio transport)
4. **Safety Evaluations**: Optional LLM-based safety checks (requires API key)
5. **Interpretability Tools**: Model analysis capabilities (requires model files)

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- ~2GB disk space for models and embeddings

## Installation Steps

### 1. Clone or navigate to the repository

```bash
cd /path/to/ai-safety-mcp-server
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -e .
```

This installs:
- `mcp` - Model Context Protocol SDK
- `sentence-transformers` - For embeddings
- `sqlmodel` - SQLite ORM
- `pydantic` - Configuration management
- Other required packages

### 4. Configure environment

```bash
cp .env.example .env
```

The default `.env` file is pre-configured for local SQLite usage:

```bash
# Vector store (SQLite with embeddings)
KB_VECTORSTORE_URL=sqlite:///./ai_safety_kb.db
KB_COLLECTION=ai_safety_docs

# Document store (leave empty - documents stored in vector store)
KB_DOCSTORE_URL=

# Safety evaluation model (optional - uses stubs if no API key)
SAFETY_EVAL_MODEL=anthropic/claude-3-opus
LITELLM_API_KEY=sk-your-api-key

# Interpretability (optional - uses stubs if not configured)
INTERP_MODEL_DIR=./models/mistral-7b

# Logging
LOG_LEVEL=INFO
MCP_TRANSPORT=stdio

# Development toggles (set to true for testing without external dependencies)
USE_KB_STUB=false
USE_EVAL_STUB=true
USE_INTERP_STUB=true
```

### 5. Populate the knowledge base

```bash
python scripts/populate_kb.py
```

This script:
- Creates the SQLite database file (`ai_safety_kb.db`)
- Generates embeddings for sample AI safety documents
- Stores 8 sample documents covering:
  - AI alignment
  - Deceptive alignment
  - Capability risks
  - Interpretability
  - Reward hacking
  - RLHF & Constitutional AI
  - Governance
  - Adversarial robustness

**First run**: Downloads the sentence-transformer model (~90MB) - this is cached for future use.

### 6. Test the knowledge base

```bash
python test_kb_search.py
```

This verifies semantic search is working correctly.

### 7. Start the MCP server

```bash
python -m mcp_server.server
```

The server runs in stdio mode, ready to receive MCP requests from clients.

## Using the Server

### Available Tools

#### Knowledge Base Tools

- **`kb.search`** - Semantic search over AI safety documents
  ```json
  {
    "query": "deceptive alignment",
    "k": 5,
    "filters": {
      "topic": "deception",
      "year_min": 2023
    }
  }
  ```

- **`kb.get_document`** - Retrieve full document by ID
  ```json
  {
    "doc_id": "deceptive-alignment"
  }
  ```

#### Safety Evaluation Tools

- **`eval.deception`** - Assess deception risk
- **`eval.dual_use`** - Assess dual-use misuse risk
- **`eval.capability_risk`** - Score dangerous capability risk
- **`eval.overall_risk`** - Aggregate risk assessment

#### Interpretability Tools

- **`interp.logit_lens`** - Inspect intermediate layer logits
- **`interp.attention`** - Inspect attention weights

#### System Tools

- **`sys.health_check`** - Report backend status

### Connecting from MCP Clients

#### Claude Desktop

Add to your Claude Desktop config:

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

#### Cursor IDE

Add as an MCP server in settings with the command:
```bash
python -m mcp_server.server
```

## Adding Your Own Documents

### Option 1: Modify the populate script

Edit `scripts/populate_kb.py` and add more documents to the `get_sample_documents()` function:

```python
Document(
    doc_id="your-doc-id",
    title="Your Document Title",
    text="Full document text...",
    url="https://example.com/your-doc",
    metadata={
        "topic": "your-topic",
        "source": "Your Source",
        "year": 2024,
        "org": "Your Org",
    },
)
```

### Option 2: Direct API usage

```python
from safety_kb.retrieval import KnowledgeBaseClient
from safety_kb.models import Document

kb = KnowledgeBaseClient(
    vectorstore_url="sqlite:///./ai_safety_kb.db",
    collection="ai_safety_docs",
    docstore_url=None,
    use_stub=False,
)

doc = Document(
    doc_id="new-doc",
    title="New Document",
    text="Document content...",
    metadata={"topic": "alignment"}
)

kb.add_document(doc)
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

All tests use in-memory stubs by default, so no database is required.

### Using Stubs for Development

Set environment variables to use in-memory stubs:

```bash
export USE_KB_STUB=true
export USE_EVAL_STUB=true
export USE_INTERP_STUB=true
```

This is useful for testing without external dependencies.

### Code Structure

```
src/
├── core/              # Configuration, logging, errors
├── mcp_server/        # MCP server implementation
├── safety_kb/         # Knowledge base (SQLite + embeddings)
├── safety_evals/      # Safety evaluation models
├── safety_interp/     # Interpretability tools
└── safety_governance/ # Governance document retrieval
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'sentence_transformers'"

Install the missing package:
```bash
pip install sentence-transformers
```

### "ValueError: Could not connect to a Chroma server"

Make sure your `.env` file uses SQLite, not ChromaDB:
```bash
KB_VECTORSTORE_URL=sqlite:///./ai_safety_kb.db
```

### "No such file: ai_safety_kb.db"

Run the populate script first:
```bash
python scripts/populate_kb.py
```

### Search returns no results

1. Check the database exists: `ls -lh ai_safety_kb.db`
2. Run the test script: `python test_kb_search.py`
3. Check embeddings model is downloaded (first run takes longer)

### Performance Issues

- **Slow first search**: The embedding model is loaded on first use (~2-3 seconds)
- **Large database**: Consider indexing optimizations or upgrading to ChromaDB
- **Memory usage**: Each embedding is ~384 dimensions × 4 bytes = ~1.5KB per document

## Advanced Configuration

### Using ChromaDB Instead

If you need better performance for large document collections:

1. Start ChromaDB server:
   ```bash
   chroma run --host localhost --port 8000
   ```

2. Update `.env`:
   ```bash
   KB_VECTORSTORE_URL=http://localhost:8000
   ```

3. Use `src/safety_kb/indexing.py` to ingest documents into ChromaDB

### Using a Different Embedding Model

Edit `src/safety_kb/retrieval.py` line 114:
```python
self._embedding_model = SentenceTransformer('your-model-name')
```

Popular alternatives:
- `all-mpnet-base-v2` (better quality, slower)
- `all-MiniLM-L12-v2` (balanced)
- `all-MiniLM-L6-v2` (default, fast)

### Separate Document Store

For very long documents, you can use a separate docstore:

```bash
KB_DOCSTORE_URL=sqlite:///./ai_safety_docs.db
```

This stores full document text separately from embeddings.

## License

Apache-2.0 (see `LICENSE`)

## Support

- File issues at: https://github.com/your-org/ai-safety-mcp-server/issues
- Documentation: See README.md
- MCP Protocol: https://modelcontextprotocol.io
