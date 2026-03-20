"""Load Fish Audio S2 Pro weights into MLX model.

Supports HuggingFace model IDs (auto-downloaded) or local paths.
Default model: mlx-community/fish-audio-s2-pro-bf16
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .model import DualARModel, SlowARConfig, FastARConfig

DEFAULT_MODEL = "mlx-community/fish-audio-s2-pro-bf16"


def resolve_model_path(model: str | Path) -> Path:
    """Resolve a HuggingFace model ID or local path to a directory.

    If `model` is an existing directory, use it directly.
    Otherwise treat it as a HF repo ID and download/cache via huggingface_hub.
    """
    path = Path(model)
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(model)))


def _map_weights(hf_weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map HuggingFace weight names to MLX model names."""
    mapped = {}

    for name, weight in hf_weights.items():
        if name.startswith("text_model.model."):
            key = name[len("text_model.model."):]
            if key == "embeddings.weight":
                key = "tok_embeddings.weight"
            mapped[f"slow.{key}"] = weight

        elif name == "audio_decoder.codebook_embeddings.weight":
            mapped["slow.codebook_embeddings.weight"] = weight

        elif name.startswith("audio_decoder."):
            key = name[len("audio_decoder."):]
            mapped[f"fast.{key}"] = weight

    return mapped


def load_model(model: str | Path = DEFAULT_MODEL,
               quantize: Optional[str] = None,
               quantize_fast: Optional[str] = None,
               quantize_slow: Optional[str] = None) -> DualARModel:
    """Load S2 Pro model from HuggingFace repo or local directory.

    Args:
        model: HF model ID or local path (default: mlx-community/fish-audio-s2-pro-bf16)
        quantize: Quantization for full model ("int4", "int8", or None)
        quantize_fast: Quantization for fast AR only
        quantize_slow: Quantization for slow AR only
    """
    model_path = resolve_model_path(model)

    with open(model_path / "config.json") as f:
        config = json.load(f)

    text_cfg = config.get("text_config", config)
    audio_cfg = config.get("audio_decoder_config", {})

    slow_config = SlowARConfig(
        dim=text_cfg.get("dim", 2560),
        n_layers=text_cfg.get("n_layer", 36),
        n_heads=text_cfg.get("n_head", 32),
        n_kv_heads=text_cfg.get("n_local_heads", 8),
        head_dim=text_cfg.get("head_dim", 128),
        intermediate_size=text_cfg.get("intermediate_size", 9728),
        vocab_size=text_cfg.get("vocab_size", 155776),
        codebook_size=audio_cfg.get("vocab_size", 4096),
        num_codebooks=audio_cfg.get("num_codebooks", 10),
        max_seq_len=text_cfg.get("max_seq_len", 32768),
        rope_base=float(text_cfg.get("rope_base", 1_000_000)),
        norm_eps=float(text_cfg.get("norm_eps", 1e-6)),
        qk_norm=text_cfg.get("attention_qk_norm", True),
        tie_word_embeddings=text_cfg.get("tie_word_embeddings", True),
        semantic_start_id=config.get("semantic_start_token_id", 151678),
    )

    fast_config = FastARConfig(
        dim=audio_cfg.get("dim", 2560),
        n_layers=audio_cfg.get("n_layer", 4),
        n_heads=audio_cfg.get("n_head", 32),
        n_kv_heads=audio_cfg.get("n_local_heads", 8),
        head_dim=audio_cfg.get("head_dim", 128),
        intermediate_size=audio_cfg.get("intermediate_size", 9728),
        codebook_size=audio_cfg.get("vocab_size", 4096),
        num_codebooks=audio_cfg.get("num_codebooks", 10),
        max_seq_len=audio_cfg.get("max_seq_len", 11),
        rope_base=float(audio_cfg.get("rope_base", 1_000_000)),
        norm_eps=float(audio_cfg.get("norm_eps", 1e-6)),
        qk_norm=audio_cfg.get("attention_qk_norm", False),
    )

    model_obj = DualARModel(slow_config, fast_config)

    weight_files = sorted(model_path.glob("model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensor files found in {model_path}")

    print(f"Loading {len(weight_files)} weight file(s)...")
    all_weights = {}
    for wf in weight_files:
        all_weights.update(mx.load(str(wf)))

    mapped = _map_weights(all_weights)
    model_obj.load_weights(list(mapped.items()))

    if quantize:
        bits = 4 if quantize == "int4" else 8
        nn.quantize(model_obj, group_size=32, bits=bits,
                     class_predicate=lambda _, m: isinstance(m, nn.Linear))
        print(f"  Model quantized to {quantize}")
    else:
        if quantize_slow:
            bits = 4 if quantize_slow == "int4" else 8
            nn.quantize(model_obj.slow, group_size=32, bits=bits,
                         class_predicate=lambda _, m: isinstance(m, nn.Linear))
            print(f"  Slow AR quantized to {quantize_slow}")
        if quantize_fast:
            bits = 4 if quantize_fast == "int4" else 8
            nn.quantize(model_obj.fast, group_size=32, bits=bits,
                         class_predicate=lambda _, m: isinstance(m, nn.Linear))
            print(f"  Fast AR quantized to {quantize_fast}")

    import mlx.utils
    n_params = sum(p.size for _, p in mlx.utils.tree_flatten(model_obj.parameters()))
    print(f"Total: {n_params / 1e9:.2f}B params")

    return model_obj
