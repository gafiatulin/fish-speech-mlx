"""Fish Audio S2 Pro — DAC codec encoder for MLX.

Encode-only path: audio waveform -> codebook indices.
Components:
  1. DAC encoder: Conv1d downsampling + Snake activations + transformer
  2. RVQ encode: downsample + pre_module transformer + VQ (semantic + residual)

All convolutions use CAUSAL padding (left-only) matching the original.
Weight-norm convolutions are pre-computed at conversion time.

Data layout: channels-last [B, T, C] throughout.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from dac_decoder import (
    DACConfig,
    Snake1d,
    CausalConv1d,
    ResidualUnit,
    ConvNeXtBlock,
    RVQTransformerBlock,
    _apply_rotary_emb,
    _conv1d,
    _prepare_conv_weights,
)


# ============================================================
# Encoder blocks
# ============================================================

class EncoderBlock(nn.Module):
    """Encoder downsampling block: 3 ResidualUnits + Snake + stride conv."""
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.res1 = ResidualUnit(input_dim, dilation=1)
        self.res2 = ResidualUnit(input_dim, dilation=3)
        self.res3 = ResidualUnit(input_dim, dilation=9)
        self.snake = Snake1d(input_dim)
        self.conv = CausalConv1d(input_dim, output_dim,
                                  kernel_size=stride * 2, bias=True)
        self._stride = stride

    def __call__(self, x):
        # x: [B, T, C] channels-last
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.snake(x)
        # Strided causal conv
        if self.conv.causal_padding > 0:
            padded = mx.pad(x, [(0, 0), (self.conv.causal_padding, 0), (0, 0)])
        else:
            padded = x
        return _conv1d(padded, self.conv.weight, self.conv.bias, stride=self._stride)


# ============================================================
# Encoder transformer (same structure as RVQ transformer)
# ============================================================

class EncoderTransformer(nn.Module):
    """WindowLimitedTransformer in encoder block.4 — 4 layers with RoPE + LayerScale."""
    def __init__(self, dim: int = 1024, n_heads: int = 16, intermediate: int = 3072,
                 n_layers: int = 4, max_len: int = 16384):
        super().__init__()
        self.layers = [
            RVQTransformerBlock(dim, n_heads, intermediate)
            for _ in range(n_layers)
        ]
        self.norm = nn.RMSNorm(dim)
        self.freqs_cis = mx.zeros((max_len, dim // n_heads // 2, 2))

    def __call__(self, x):
        """x: [B, T, C] channels-last."""
        T = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(x.dtype)
        for layer in self.layers:
            x = layer(x, self.freqs_cis, mask=mask)
        x = self.norm(x)
        return x


# ============================================================
# DAC Encoder
# ============================================================

class DACEncoder(nn.Module):
    """DAC encoder: audio waveform -> latent vectors. Channels-last."""
    def __init__(self, config: DACConfig = DACConfig()):
        super().__init__()
        d = config.encoder_dim  # 64

        # block.0: input conv
        self.input_conv = CausalConv1d(1, d, kernel_size=7)

        # block.1-4: encoder blocks
        self.blocks = []
        channels = [d]  # [64]
        for rate in config.encoder_rates:  # (2, 4, 8, 8)
            out_d = d * 2
            channels.append(out_d)
            self.blocks.append(EncoderBlock(d, out_d, stride=rate))
            d = out_d
        # channels = [64, 128, 256, 512, 1024]

        # block.4 also has transformer after the stride conv
        self.encoder_transformer = EncoderTransformer(
            dim=d, n_heads=16, intermediate=3072, n_layers=4, max_len=16384
        )

        # block.5: output snake
        self.output_snake = Snake1d(d)

        # block.6: output conv
        self.output_conv = CausalConv1d(d, config.d_latent, kernel_size=3)

    def __call__(self, audio: mx.array) -> mx.array:
        """Encode audio to latent.

        Args:
            audio: [B, 1, T], [B, T], or [B, T, 1] waveform at 44.1kHz

        Returns:
            z: [B, T', d_latent] latent representation (channels-last)
        """
        if audio.ndim == 2:
            audio = audio[:, :, None]       # [B, T] -> [B, T, 1]
        elif audio.shape[1] == 1:
            audio = audio.transpose(0, 2, 1)  # [B, 1, T] -> [B, T, 1]

        x = self.input_conv(audio)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 3:  # After block.4's stride conv, apply transformer
                x = self.encoder_transformer(x)
        x = self.output_snake(x)
        x = self.output_conv(x)
        return x


# ============================================================
# Downsample stage (quantizer.downsample)
# ============================================================

class DownsampleStage(nn.Module):
    """Downsample: CausalConv1d stride-2 + ConvNeXt block.

    Note: unlike upsample, downsample uses strided Conv1d (not transposed).
    The convs here are NOT weight-normed (plain weights), but still use causal padding.
    Weights stored in PyTorch format, transposed by prepare_weights().
    """
    def __init__(self, dim: int, factor: int):
        super().__init__()
        # Stored as [out, in, kernel] (PyTorch); prepare_weights transposes to [out, kernel, in]
        self.conv_weight = mx.zeros((dim, dim, factor))
        self.conv_bias = mx.zeros((dim,))
        self._factor = factor
        self._causal_padding = factor - 1  # left-pad by kernel_size - 1
        self.cnx = ConvNeXtBlock(dim)

    def __call__(self, x):
        # x: [B, T, C] channels-last
        if self._causal_padding > 0:
            x = mx.pad(x, [(0, 0), (self._causal_padding, 0), (0, 0)])
        y = _conv1d(x, self.conv_weight, self.conv_bias, stride=self._factor)
        y = self.cnx(y)
        return y


# ============================================================
# RVQ Pre-module (same as post-module)
# ============================================================

class RVQPreModule(nn.Module):
    """RVQ pre-quantization transformer with RoPE + LayerScale. Channels-last I/O."""
    def __init__(self, config: DACConfig):
        super().__init__()
        dim = config.rvq_transformer_dim
        self.layers = [
            RVQTransformerBlock(dim, config.rvq_transformer_heads,
                                config.rvq_transformer_intermediate)
            for _ in range(config.rvq_transformer_layers)
        ]
        self.norm = nn.RMSNorm(dim)
        self.freqs_cis = mx.zeros((4096, dim // config.rvq_transformer_heads // 2, 2))

    def __call__(self, x):
        """x: [B, T, C] channels-last."""
        T = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(x.dtype)
        for layer in self.layers:
            x = layer(x, self.freqs_cis, mask=mask)
        x = self.norm(x)
        return x


# ============================================================
# Vector Quantize (encode path only)
# ============================================================

class VectorQuantize(nn.Module):
    """Single VQ codebook: in_proj -> L2 normalize -> nearest neighbor."""
    def __init__(self, d_latent: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        # in_proj: Conv1d(d_latent, codebook_dim, k=1) — weight-normed, pre-computed
        self.in_proj_weight = mx.zeros((codebook_dim, d_latent))  # [8, 1024]
        self.in_proj_bias = mx.zeros((codebook_dim,))
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def encode(self, z: mx.array) -> mx.array:
        """Encode latent to codebook indices.

        Args:
            z: [B, T, d_latent] latent (channels-last)

        Returns:
            indices: [B, T] codebook indices
        """
        projected = z @ self.in_proj_weight.T + self.in_proj_bias  # [B, T, codebook_dim]

        # L2 normalize
        projected = projected / (mx.linalg.norm(projected, axis=-1, keepdims=True) + 1e-12)

        # Nearest neighbor in codebook (also L2 normalized)
        cb = self.codebook.weight  # [K, codebook_dim]
        cb_norm = cb / (mx.linalg.norm(cb, axis=-1, keepdims=True) + 1e-12)

        # Maximize dot product = minimize L2 distance (both unit-normed)
        dots = projected @ cb_norm.T  # [B, T, K]
        indices = mx.argmax(dots, axis=-1)  # [B, T]
        return indices

    def decode_codes(self, indices: mx.array) -> mx.array:
        """Decode indices back to latent contribution (for residual VQ)."""
        return self.codebook(indices)  # [B, T, codebook_dim]


# ============================================================
# RVQ Encode Path
# ============================================================

class RVQEncode(nn.Module):
    """RVQ encode path: latent -> codebook indices."""
    def __init__(self, config: DACConfig = DACConfig()):
        super().__init__()
        self.config = config

        # Downsample
        self.downsample = [
            DownsampleStage(config.d_latent, factor)
            for factor in config.downsample_factor
        ]

        # Pre-module transformer
        self.pre_module = RVQPreModule(config)

        # Semantic VQ (1 codebook, 4096 entries)
        self.semantic_vq = VectorQuantize(
            config.d_latent, config.codebook_size_semantic, config.codebook_dim
        )
        # Semantic out_proj for residual computation
        self.semantic_out_proj_weight = mx.zeros((config.d_latent, config.codebook_dim))
        self.semantic_out_proj_bias = mx.zeros((config.d_latent,))

        # Residual VQs (9 codebooks, 1024 entries each)
        self.residual_vqs = [
            VectorQuantize(config.d_latent, config.codebook_size_residual, config.codebook_dim)
            for _ in range(config.n_residual_codebooks)
        ]
        self.residual_out_proj_weights = [
            mx.zeros((config.d_latent, config.codebook_dim))
            for _ in range(config.n_residual_codebooks)
        ]
        self.residual_out_proj_biases = [
            mx.zeros((config.d_latent,))
            for _ in range(config.n_residual_codebooks)
        ]

    def __call__(self, z: mx.array) -> mx.array:
        """Encode latent to all codebook indices.

        Args:
            z: [B, T', d_latent] latent representation (channels-last)

        Returns:
            codes: [B, 10, T''] codebook indices (semantic + 9 residual)
        """
        # Downsample
        for stage in self.downsample:
            z = stage(z)

        # Pre-module transformer
        z = self.pre_module(z)

        all_indices = []

        # Semantic quantization
        sem_indices = self.semantic_vq.encode(z)  # [B, T]
        all_indices.append(sem_indices)

        # Compute residual: z - out_proj(codebook[indices])
        sem_codes = self.semantic_vq.codebook(sem_indices)  # [B, T, 8]
        sem_decoded = sem_codes @ self.semantic_out_proj_weight.T + self.semantic_out_proj_bias  # [B, T, 1024]
        residual = z - sem_decoded  # [B, T, 1024] — no transpose needed

        # Residual quantization
        for i in range(self.config.n_residual_codebooks):
            res_indices = self.residual_vqs[i].encode(residual)  # [B, T]
            all_indices.append(res_indices)

            # Update residual
            res_codes = self.residual_vqs[i].codebook(res_indices)  # [B, T, 8]
            res_decoded = (res_codes @ self.residual_out_proj_weights[i].T
                          + self.residual_out_proj_biases[i])  # [B, T, 1024]
            residual = residual - res_decoded  # No transpose needed

        codes = mx.stack(all_indices, axis=1)  # [B, 10, T']
        return codes


# ============================================================
# Full codec encode pipeline
# ============================================================

class FishCodecEncoder(nn.Module):
    """Complete encode pipeline: audio -> codes."""
    def __init__(self, config: DACConfig = DACConfig()):
        super().__init__()
        self.config = config
        self.encoder = DACEncoder(config)
        self.rvq_enc = RVQEncode(config)

    def prepare_weights(self):
        """Transpose conv weights from PyTorch to MLX format. Call once after load_weights()."""
        _prepare_conv_weights(self)
        # Also transpose DownsampleStage conv_weight (not a CausalConv1d module)
        for stage in self.rvq_enc.downsample:
            if stage.conv_weight.ndim == 3:
                stage.conv_weight = stage.conv_weight.transpose(0, 2, 1)

    def __call__(self, audio: mx.array) -> mx.array:
        """Encode audio waveform to codebook indices.

        Args:
            audio: [B, T] or [B, 1, T] waveform at 44.1kHz

        Returns:
            codes: [B, 10, T'] codebook indices
        """
        z = self.encoder(audio)
        codes = self.rvq_enc(z)
        return codes


# ============================================================
# Encoder weight mapping (raw PyTorch → MLX model names)
# ============================================================

def _map_raw_encoder_residual_unit(state, mapped, src_prefix, dst_prefix):
    """Map a raw ResidualUnit for encoder."""
    from dac_decoder import _wn_weight
    mapped[f"{dst_prefix}.snake1.alpha"] = state[f"{src_prefix}.0.alpha"]
    w = _wn_weight(state[f"{src_prefix}.1.weight_g"], state[f"{src_prefix}.1.weight_v"])
    mapped[f"{dst_prefix}.conv1.weight"] = w
    mapped[f"{dst_prefix}.conv1.bias"] = state[f"{src_prefix}.1.bias"]
    mapped[f"{dst_prefix}.snake2.alpha"] = state[f"{src_prefix}.2.alpha"]
    w = _wn_weight(state[f"{src_prefix}.3.weight_g"], state[f"{src_prefix}.3.weight_v"])
    mapped[f"{dst_prefix}.conv2.weight"] = w
    mapped[f"{dst_prefix}.conv2.bias"] = state[f"{src_prefix}.3.bias"]


def _map_raw_codec_encode(state):
    """Map raw PyTorch encoder weights to MLX model names."""
    from dac_decoder import _wn_weight, _map_transformer_layers, _map_convnext
    mapped = {}

    # 1. DAC Encoder input conv
    w = _wn_weight(state["encoder.block.0.weight_g"], state["encoder.block.0.weight_v"])
    mapped["encoder.input_conv.weight"] = w
    mapped["encoder.input_conv.bias"] = state["encoder.block.0.bias"]

    # 2. Encoder blocks (4x)
    for block_idx in range(4):
        sp = f"encoder.block.{block_idx + 1}"
        dp = f"encoder.blocks.{block_idx}"

        for res_idx, res_name in enumerate(["res1", "res2", "res3"]):
            _map_raw_encoder_residual_unit(state, mapped,
                                           f"{sp}.block.{res_idx}.block",
                                           f"{dp}.{res_name}")

        mapped[f"{dp}.snake.alpha"] = state[f"{sp}.block.3.alpha"]
        w = _wn_weight(state[f"{sp}.block.4.weight_g"], state[f"{sp}.block.4.weight_v"])
        mapped[f"{dp}.conv.weight"] = w
        mapped[f"{dp}.conv.bias"] = state[f"{sp}.block.4.bias"]

    # 3. Encoder transformer (block.4.block.5)
    _map_transformer_layers(state, mapped,
                            "encoder.block.4.block.5",
                            "encoder.encoder_transformer", 4)

    # 4. Output snake + conv
    mapped["encoder.output_snake.alpha"] = state["encoder.block.5.alpha"]
    w = _wn_weight(state["encoder.block.6.weight_g"], state["encoder.block.6.weight_v"])
    mapped["encoder.output_conv.weight"] = w
    mapped["encoder.output_conv.bias"] = state["encoder.block.6.bias"]

    # 5. Downsample (2 stages, NOT weight-normed)
    for i in range(2):
        sp = f"quantizer.downsample.{i}"
        dp = f"rvq_enc.downsample.{i}"
        mapped[f"{dp}.conv_weight"] = state[f"{sp}.0.conv.weight"]
        mapped[f"{dp}.conv_bias"] = state[f"{sp}.0.conv.bias"]
        _map_convnext(state, mapped, f"{sp}.1", f"{dp}.cnx")

    # 6. Pre-module transformer (8 layers)
    _map_transformer_layers(state, mapped,
                            "quantizer.pre_module",
                            "rvq_enc.pre_module", 8)

    # 7. VQ in_projs + codebooks + out_projs
    prefix = "quantizer.semantic_quantizer.quantizers.0"
    w = _wn_weight(state[f"{prefix}.in_proj.weight_g"], state[f"{prefix}.in_proj.weight_v"])
    mapped["rvq_enc.semantic_vq.in_proj_weight"] = w.squeeze(-1)
    mapped["rvq_enc.semantic_vq.in_proj_bias"] = state[f"{prefix}.in_proj.bias"]
    mapped["rvq_enc.semantic_vq.codebook.weight"] = state[f"{prefix}.codebook.weight"]
    w = _wn_weight(state[f"{prefix}.out_proj.weight_g"], state[f"{prefix}.out_proj.weight_v"])
    mapped["rvq_enc.semantic_out_proj_weight"] = w.squeeze(-1)
    mapped["rvq_enc.semantic_out_proj_bias"] = state[f"{prefix}.out_proj.bias"]

    for i in range(9):
        prefix = f"quantizer.quantizer.quantizers.{i}"
        w = _wn_weight(state[f"{prefix}.in_proj.weight_g"], state[f"{prefix}.in_proj.weight_v"])
        mapped[f"rvq_enc.residual_vqs.{i}.in_proj_weight"] = w.squeeze(-1)
        mapped[f"rvq_enc.residual_vqs.{i}.in_proj_bias"] = state[f"{prefix}.in_proj.bias"]
        mapped[f"rvq_enc.residual_vqs.{i}.codebook.weight"] = state[f"{prefix}.codebook.weight"]
        w_out = _wn_weight(state[f"{prefix}.out_proj.weight_g"], state[f"{prefix}.out_proj.weight_v"])
        mapped[f"rvq_enc.residual_out_proj_weights.{i}"] = w_out.squeeze(-1)
        mapped[f"rvq_enc.residual_out_proj_biases.{i}"] = state[f"{prefix}.out_proj.bias"]

    return mapped


def load_codec_encoder_weights(codec_enc, weights: dict):
    """Load encoder weights, auto-detecting format (pre-converted or raw PyTorch)."""
    enc_weights = {k: v for k, v in weights.items()
                   if k.startswith("encoder.") or k.startswith("rvq_enc.")
                   or k.startswith("quantizer.")}

    is_raw = any("weight_g" in k for k in enc_weights)

    if is_raw:
        mapped = _map_raw_codec_encode(enc_weights)
    else:
        # Pre-converted: filter to encoder-only keys
        mapped = {k: v for k, v in weights.items()
                  if k.startswith("encoder.") or k.startswith("rvq_enc.")}

    codec_enc.load_weights(list(mapped.items()))
    codec_enc.prepare_weights()
