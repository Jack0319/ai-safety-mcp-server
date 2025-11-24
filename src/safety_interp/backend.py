"""Shared interpretability backend utilities."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.errors import UnsupportedModelError
from core.logging import get_logger

logger = get_logger(__name__)


class InterpretabilityBackend:
    def __init__(self, model_dir: Optional[Union[Path, str]], use_stub: bool = False):
        """
        Initialize interpretability backend.
        
        Args:
            model_dir: Can be:
                - Local path (Path or str): e.g., "/path/to/model" or "./models/gpt2"
                - HuggingFace model ID (str): e.g., "gpt2", "mistralai/Mistral-7B-v0.1"
            use_stub: If True, use stub mode (no actual model loading)
        """
        self.use_stub = use_stub
        
        # Handle model_dir - can be Path, local path string, or HuggingFace model ID
        if model_dir is None:
            self.model_dir = None
            self.model_id = None
            self.use_stub = True
        elif isinstance(model_dir, Path):
            model_path_str = str(model_dir)
            # Check if it's a local path that exists
            if model_dir.exists() and model_dir.is_dir():
                self.model_dir = model_dir
                self.model_id = str(model_dir)
            else:
                # Treat as HuggingFace model ID
                self.model_dir = None
                self.model_id = model_path_str
        else:
            # String - could be local path or HuggingFace ID
            model_path = Path(model_dir)
            if model_path.exists() and model_path.is_dir():
                # Local path
                self.model_dir = model_path
                self.model_id = str(model_path)
            else:
                # HuggingFace model ID
                self.model_dir = None
                self.model_id = model_dir
        
        if self.use_stub:
            self.model_id = "stub"
        
        self._artifacts: Tuple | None = None
        if self.use_stub:
            logger.info("Interpretability backend initialized in stub mode (no model loaded)")
        elif not self.model_id:
            logger.warning("Interpretability backend initialized without model configuration (will use stub mode)")
        else:
            logger.info(f"Interpretability backend initialized: model_id='{self.model_id}' (lazy loading - model will load on first tool call)")
            # Note: Model will be loaded lazily on first access to avoid startup delays

    @property
    def tokenizer(self):  # type: ignore[override]
        if not self._artifacts:
            if not self.use_stub and self.model_id:
                logger.info(f"Lazy loading model '{self.model_id}' (first access to tokenizer)")
                try:
                    self._artifacts = _load_artifacts(self.model_id, self.model_dir)
                except Exception as e:
                    logger.error(f"Failed to lazy load model '{self.model_id}': {e}")
                    raise UnsupportedModelError(
                        f"Model not loaded. Configured model: {self.model_id}. "
                        f"Set INTERP_MODEL_DIR to a local path or HuggingFace model ID, or set USE_INTERP_STUB=true"
                    ) from e
            else:
                raise UnsupportedModelError(
                    f"Model not loaded. Configured model: {self.model_id}. "
                    f"Set INTERP_MODEL_DIR to a local path or HuggingFace model ID, or set USE_INTERP_STUB=true"
                )
        return self._artifacts[0]

    @property
    def model(self):  # type: ignore[override]
        if not self._artifacts:
            if not self.use_stub and self.model_id:
                logger.info(f"Lazy loading model '{self.model_id}' (first access to model)")
                try:
                    self._artifacts = _load_artifacts(self.model_id, self.model_dir)
                except Exception as e:
                    logger.error(f"Failed to lazy load model '{self.model_id}': {e}")
                    raise UnsupportedModelError(
                        f"Model not loaded. Configured model: {self.model_id}. "
                        f"Set INTERP_MODEL_DIR to a local path or HuggingFace model ID, or set USE_INTERP_STUB=true"
                    ) from e
            else:
                raise UnsupportedModelError(
                    f"Model not loaded. Configured model: {self.model_id}. "
                    f"Set INTERP_MODEL_DIR to a local path or HuggingFace model ID, or set USE_INTERP_STUB=true"
                )
        return self._artifacts[1]

    def ensure_layer_index(self, layer_index: int) -> int:
        if self.use_stub:
            return max(0, layer_index)
        num_layers = self.model.config.num_hidden_layers
        if layer_index < 0 or layer_index >= num_layers:
            raise UnsupportedModelError(f"Layer index {layer_index} out of range (0-{num_layers - 1})")
        return layer_index


@lru_cache(maxsize=2)
def _load_artifacts(model_id: str, model_dir: Optional[Path] = None):
    """
    Load model and tokenizer (lazy loading - called on first access).
    
    Args:
        model_id: HuggingFace model ID (e.g., "gpt2") or local path string
        model_dir: Optional local Path (if None, model_id is treated as HuggingFace ID)
    """
    import time
    start_time = time.time()
    
    try:
        if model_dir:
            # Load from local path
            logger.info(f"Loading model from local path: {model_dir}")
            load_path = str(model_dir)
        else:
            # Load from HuggingFace
            logger.info(f"Loading model from HuggingFace: {model_id} (this may take a while on first download)")
            load_path = model_id
        
        logger.info(f"Downloading/loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            output_hidden_states=True,
            output_attentions=True,
        )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        elapsed = time.time() - start_time
        logger.info(f"Successfully loaded model '{model_id}' in {elapsed:.2f}s (cached for future use)")
        return tokenizer, model
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error loading model '{model_id}' after {elapsed:.2f}s: {e}", exc_info=True)
        raise
