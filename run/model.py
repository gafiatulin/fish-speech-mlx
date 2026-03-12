"""Fish Audio S2 Pro — MLX model definitions.

Implements the Dual-AR architecture:
  - Slow AR: 36-layer Qwen3 (4B) predicting semantic tokens
  - Fast AR: 4-layer decoder (400M) predicting residual codebook tokens
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class SlowARConfig:
    """Slow AR (text model) config — Qwen3-style."""
    dim: int = 2560
    n_layers: int = 36
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    vocab_size: int = 155776
    codebook_size: int = 4096
    num_codebooks: int = 10
    max_seq_len: int = 32768
    rope_base: float = 1_000_000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_word_embeddings: bool = True
    scale_codebook_embeddings: bool = True
    norm_fastlayer_input: bool = True
    semantic_start_id: int = 151678


@dataclass
class FastARConfig:
    """Fast AR (audio decoder) config."""
    dim: int = 2560
    n_layers: int = 4
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    codebook_size: int = 4096
    num_codebooks: int = 10
    max_seq_len: int = 11
    rope_base: float = 1_000_000.0
    norm_eps: float = 1e-6
    qk_norm: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class KVCache:
    """Pre-allocated KV cache for Slow AR.

    Avoids O(n²) memory traffic from per-step concatenation by writing
    into fixed-size buffers. Grows by 2x when capacity is exceeded.
    """
    def __init__(self, n_kv_heads: int = 0, head_dim: int = 0,
                 max_len: int = 4096):
        self.offset: int = 0
        if n_kv_heads > 0 and head_dim > 0:
            shape = (1, n_kv_heads, max_len, head_dim)
            self.keys = mx.zeros(shape, dtype=mx.bfloat16)
            self.values = mx.zeros(shape, dtype=mx.bfloat16)
            self._capacity = max_len
        else:
            self.keys: Optional[mx.array] = None
            self.values: Optional[mx.array] = None
            self._capacity = 0

    def _ensure_capacity(self, needed: int):
        """Double buffer size until it fits `needed` total entries."""
        if self._capacity >= needed:
            return
        new_cap = max(self._capacity * 2, needed)
        B, H, _, D = self.keys.shape
        new_k = mx.zeros((B, H, new_cap, D), dtype=self.keys.dtype)
        new_v = mx.zeros((B, H, new_cap, D), dtype=self.values.dtype)
        if self.offset > 0:
            new_k[:, :, :self.offset, :] = self.keys[:, :, :self.offset, :]
            new_v[:, :, :self.offset, :] = self.values[:, :, :self.offset, :]
        self.keys = new_k
        self.values = new_v
        self._capacity = new_cap

    def update(self, k: mx.array, v: mx.array) -> Tuple[mx.array, mx.array]:
        # k, v: [1, n_kv_heads, L, head_dim]
        L = k.shape[2]
        end = self.offset + L

        if self.keys is None:
            # Lazy init from first update (fallback for unparameterized construction)
            self.keys = k
            self.values = v
            self._capacity = L
        elif end > self._capacity:
            self._ensure_capacity(end)
            self.keys[:, :, self.offset:end, :] = k
            self.values[:, :, self.offset:end, :] = v
        else:
            self.keys[:, :, self.offset:end, :] = k
            self.values[:, :, self.offset:end, :] = v

        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def set_from(self, keys: mx.array, values: mx.array):
        """Restore cache from saved state (e.g. voice cache)."""
        seq_len = keys.shape[2]
        # Allocate with headroom for generation
        cap = max(seq_len + 2048, seq_len * 2)
        B, H, _, D = keys.shape
        self.keys = mx.zeros((B, H, cap, D), dtype=keys.dtype)
        self.values = mx.zeros((B, H, cap, D), dtype=values.dtype)
        self.keys[:, :, :seq_len, :] = keys
        self.values[:, :, :seq_len, :] = values
        self.offset = seq_len
        self._capacity = cap

    def reset(self):
        self.keys = None
        self.values = None
        self.offset = 0
        self._capacity = 0


class RotatingKVCache:
    """Pre-allocated KV cache for short sequences (Fast AR).

    Avoids repeated concatenation by writing into fixed-size buffers.
    """
    def __init__(self, n_kv_heads: int, head_dim: int, max_len: int = 16):
        self.max_len = max_len
        self.offset = 0
        # Pre-allocate [1, n_kv_heads, max_len, head_dim]
        shape = (1, n_kv_heads, max_len, head_dim)
        self.keys = mx.zeros(shape, dtype=mx.bfloat16)
        self.values = mx.zeros(shape, dtype=mx.bfloat16)

    def update(self, k: mx.array, v: mx.array) -> Tuple[mx.array, mx.array]:
        # k, v: [1, n_kv_heads, L, head_dim]
        L = k.shape[2]
        end = self.offset + L
        self.keys[:, :, self.offset:end, :] = k
        self.values[:, :, self.offset:end, :] = v
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def reset(self):
        self.offset = 0


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, qk_norm: bool = False, norm_eps: float = 1e-6,
                 rope_base: float = 1_000_000.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        total_head_dim = (n_heads + 2 * n_kv_heads) * head_dim
        self.wqkv = nn.Linear(dim, total_head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=norm_eps)

        self.rope = nn.RoPE(head_dim, traditional=True, base=rope_base)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 cache=None):
        B, L, _ = x.shape

        qkv = self.wqkv(x)
        q_dim = self.n_heads * self.head_dim
        k_dim = self.n_kv_heads * self.head_dim

        q = qkv[:, :, :q_dim].reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = qkv[:, :, q_dim:q_dim + k_dim].reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = qkv[:, :, q_dim + k_dim:].reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update(k, v)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, intermediate_size: int,
                 qk_norm: bool = False, norm_eps: float = 1e-6,
                 rope_base: float = 1_000_000.0):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads, head_dim, qk_norm, norm_eps, rope_base)
        self.feed_forward = FeedForward(dim, intermediate_size)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 cache=None):
        h = x + self.attention(self.attention_norm(x), mask=mask, cache=cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class SlowAR(nn.Module):
    """Slow AR: 36-layer Qwen3 that predicts semantic tokens."""

    def __init__(self, config: SlowARConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks, config.dim
        )

        self.layers = [
            TransformerBlock(
                config.dim, config.n_heads, config.n_kv_heads,
                config.head_dim, config.intermediate_size,
                qk_norm=config.qk_norm, norm_eps=config.norm_eps,
                rope_base=config.rope_base,
            )
            for _ in range(config.n_layers)
        ]
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if not config.tie_word_embeddings:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def embed(self, tokens: mx.array, codebooks: Optional[mx.array] = None) -> mx.array:
        """Embed tokens with optional codebook sum.

        Args:
            tokens: [B, L] vocab token IDs
            codebooks: [B, L, num_codebooks] raw codebook indices (0-4095), or None
        """
        x = self.tok_embeddings(tokens)

        if codebooks is not None:
            cfg = self.config
            is_semantic = (tokens >= cfg.semantic_start_id) & (
                tokens < cfg.semantic_start_id + cfg.codebook_size
            )

            # Batched codebook embedding: single lookup instead of 10 sequential ones
            offsets = mx.arange(cfg.num_codebooks) * cfg.codebook_size  # [10]
            all_indices = codebooks + offsets[None, None, :]  # [B, L, 10]
            all_embeds = self.codebook_embeddings(all_indices)  # [B, L, 10, dim]
            cb_sum = all_embeds.sum(axis=2)  # [B, L, dim]

            mask = is_semantic[:, :, None].astype(x.dtype)
            cb_sum = cb_sum * mask
            x = x + cb_sum

            if cfg.scale_codebook_embeddings:
                x = mx.where(mask > 0, x / math.sqrt(cfg.num_codebooks + 1), x)

        return x

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 cache=None, output_weights: Optional[mx.array] = None):
        for i, layer in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            x = layer(x, mask=mask, cache=c)

        h = self.norm(x)

        if output_weights is not None:
            # Projected output: compute logits only for pre-selected tokens
            logits = h @ output_weights.T
        elif self.config.tie_word_embeddings:
            logits = h @ self.tok_embeddings.weight.T
        else:
            logits = self.output(h)

        hidden_out = h if self.config.norm_fastlayer_input else x
        return logits, hidden_out


class FastAR(nn.Module):
    """Fast AR: 4-layer decoder that fills residual codebooks."""

    def __init__(self, config: FastARConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.codebook_size, config.dim)

        self.layers = [
            TransformerBlock(
                config.dim, config.n_heads, config.n_kv_heads,
                config.head_dim, config.intermediate_size,
                qk_norm=config.qk_norm, norm_eps=config.norm_eps,
                rope_base=config.rope_base,
            )
            for _ in range(config.n_layers)
        ]
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.codebook_size, bias=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 cache=None):
        for i, layer in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            x = layer(x, mask=mask, cache=c)

        h = self.norm(x)
        logits = self.output(h)
        return logits


class DualARModel(nn.Module):
    """Complete Fish Audio S2 Pro dual-AR model."""

    def __init__(self, slow_config: SlowARConfig, fast_config: FastARConfig):
        super().__init__()
        self.slow = SlowAR(slow_config)
        self.fast = FastAR(fast_config)

    @property
    def slow_config(self):
        return self.slow.config

    @property
    def fast_config(self):
        return self.fast.config
