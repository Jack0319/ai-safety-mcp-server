# Model Configuration Guide

## Interpretability Models (`interp.*` tools)

### Supported Model Formats

The `INTERP_MODEL_DIR` environment variable accepts:

1. **HuggingFace Model ID** (recommended for quick start):
   ```bash
   INTERP_MODEL_DIR=gpt2
   # or
   INTERP_MODEL_DIR=mistralai/Mistral-7B-v0.1
   # or
   INTERP_MODEL_DIR=meta-llama/Llama-2-7b-hf
   ```

2. **Local Path** (for downloaded models):
   ```bash
   INTERP_MODEL_DIR=/path/to/local/model
   # or
   INTERP_MODEL_DIR=./models/gpt2
   ```

### How It Works

- The backend automatically detects whether the value is a local path or HuggingFace model ID
- If the path exists locally, it loads from disk
- If not, it treats it as a HuggingFace model ID and downloads it (cached by HuggingFace)
- Models are loaded lazily on first tool call and cached in memory

### Examples

**Using HuggingFace model ID:**
```bash
export INTERP_MODEL_DIR=gpt2
python -m mcp_server.server
```

**Using local model:**
```bash
# First download the model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('gpt2')"

# Then use local path
export INTERP_MODEL_DIR=~/.cache/huggingface/hub/models--gpt2/snapshots/...
python -m mcp_server.server
```

**Stub mode (no model needed):**
```bash
export USE_INTERP_STUB=true
# INTERP_MODEL_DIR not needed
python -m mcp_server.server
```

## Safety Evaluation Models (`eval.*` tools)

### Why LiteLLM?

LiteLLM is **ONLY needed for `eval.*` tools**, NOT for `interp.*` tools.

**Purpose:**
- Provides a unified interface to call various LLM APIs (OpenAI, Anthropic, Google, etc.)
- Used for safety evaluations that require LLM-based classification
- Not needed for interpretability tools (which use local models)

**Configuration:**
```bash
LITELLM_API_KEY=your_api_key_here
SAFETY_EVAL_MODEL=anthropic/claude-3-opus
```

**Without LiteLLM:**
- Set `USE_EVAL_STUB=true` to use heuristic-based evaluations
- No API key needed
- Less accurate but works offline

### Supported Models

Any model supported by LiteLLM:
- OpenAI: `gpt-4`, `gpt-3.5-turbo`
- Anthropic: `anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`
- Google: `gemini/gemini-pro`
- And many more via LiteLLM

## Troubleshooting

### Error: "models/mistral-7b is not a local folder"

**Problem:** The path doesn't exist locally and isn't a valid HuggingFace model ID.

**Solutions:**
1. Use a valid HuggingFace model ID: `INTERP_MODEL_DIR=mistralai/Mistral-7B-v0.1`
2. Use a local path that exists: `INTERP_MODEL_DIR=/actual/path/to/model`
3. Use stub mode: `USE_INTERP_STUB=true`

### Error: "Failed to load model"

**Check:**
- Model ID is valid (check https://huggingface.co/models)
- Internet connection (for HuggingFace downloads)
- Disk space (models can be large)
- Memory (models need RAM/VRAM)

### Model Loading Takes Too Long

- First load downloads the model (can be GB)
- Subsequent loads use cached model
- Consider using smaller models for testing (e.g., `gpt2` instead of `mistralai/Mistral-7B-v0.1`)

