"""Fish Audio S2 Pro — DAC codec decoder for MLX.

Decode-only path: codebook indices -> audio waveform.
Components:
  1. RVQ decode: codebook lookup + post_module transformer + upsample
  2. DAC decoder: ConvTranspose1d upsampling + Snake activations

All convolutions use CAUSAL padding (left-only) matching the original
CausalConvNet / CausalTransConvNet implementations.
Weight-norm convolutions are pre-computed at conversion time.

Data layout: channels-last [B, T, C] throughout to avoid per-call transposes.
Weights are stored in PyTorch format [out, in, kernel] on disk and transposed
to MLX format [out, kernel, in] once after loading via prepare_weights().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DACConfig:
    """DAC codec configuration."""
    sample_rate: int = 44100
    encoder_dim: int = 64
    encoder_rates: tuple = (2, 4, 8, 8)
    decoder_dim: int = 1536
    decoder_rates: tuple = (8, 8, 4, 2)
    d_latent: int = 1024
    # RVQ
    n_semantic_codebooks: int = 1
    n_residual_codebooks: int = 9
    codebook_size_semantic: int = 4096
    codebook_size_residual: int = 1024
    codebook_dim: int = 8
    downsample_factor: tuple = (2, 2)
    # Transformers
    rvq_transformer_layers: int = 8
    rvq_transformer_heads: int = 16
    rvq_transformer_dim: int = 1024
    rvq_transformer_intermediate: int = 3072


# ============================================================
# Activations
# ============================================================

# Fused Snake1d Metal kernel: x + sin²(αx) / α in a single pass.
# Input x is [B, T, C] channels-last; alpha is [C] (squeezed from [1,1,C]).
# Avoids 4 separate elementwise ops + 1 sin by fusing into one kernel.
_snake1d_kernel = mx.fast.metal_kernel(
    name="snake1d",
    input_names=["x", "alpha"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        uint C = alpha_shape[0];
        uint c = elem % C;
        T a = alpha[c];
        T val = x[elem];
        T s = metal::sin(a * val);
        out[elem] = val + (s * s) / (a + T(1e-9));
    """,
)


def _snake1d_fused(x: mx.array, alpha: mx.array) -> mx.array:
    """Apply fused Snake1d activation via custom Metal kernel."""
    # Squeeze alpha to 1D [C] for kernel indexing
    alpha_1d = alpha.reshape(-1)
    outputs = _snake1d_kernel(
        inputs=[x, alpha_1d],
        template=[("T", x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(min(256, x.size), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return outputs[0]


class Snake1d(nn.Module):
    """Snake activation: x + (1/alpha) * sin^2(alpha * x).

    Operates on channels-last [B, T, C] tensors.
    Uses a fused Metal kernel to avoid 4 separate elementwise ops.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Shape [1, C, 1] matches PyTorch saved format for weight loading;
        # prepare_weights() transposes to [1, 1, C] for channels-last ops
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, x):
        return _snake1d_fused(x, self.alpha)


# ============================================================
# Causal convolution building blocks (channels-last)
# ============================================================

def _conv1d(x, weight, bias, stride=1, dilation=1, groups=1):
    """Conv1d on channels-last [B, T, C] tensors.

    Weight: [out, kernel, in/groups] (MLX native format).
    """
    y = mx.conv1d(x, weight, stride=stride, padding=0,
                  dilation=dilation, groups=groups)
    if bias is not None:
        y = y + bias
    return y


def _conv_transpose1d(x, weight, bias, stride=1):
    """ConvTranspose1d on channels-last [B, T, C].

    Weight: [out, kernel, in] (MLX native format).
    """
    y = mx.conv_transpose1d(x, weight, stride=stride, padding=0)
    if bias is not None:
        y = y + bias
    return y


class CausalConv1d(nn.Module):
    """Causal Conv1d: left-pad by (kernel-1)*dilation, no right padding.

    Data: channels-last [B, T, C].
    Weight stored as [out, in/groups, kernel] (PyTorch format) for load
    compatibility, transposed to [out, kernel, in/groups] by prepare_weights().
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        self.dilation = dilation
        self.groups = groups
        self.causal_padding = (kernel_size - 1) * dilation
        scale = math.sqrt(2.0 / (in_channels // groups * kernel_size))
        # Stored in PyTorch format; prepare_weights() transposes to MLX format
        self.weight = mx.random.normal((out_channels, in_channels // groups, kernel_size)) * scale
        self.bias = mx.zeros((out_channels,)) if bias else None
        self._prepared = False

    def __call__(self, x):
        # x: [B, T, C]
        if self.causal_padding > 0:
            x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        return _conv1d(x, self.weight, self.bias, dilation=self.dilation,
                       groups=self.groups)


class CausalConvTranspose1d(nn.Module):
    """Causal ConvTranspose1d: no padding, trim right by (kernel-stride).

    Data: channels-last [B, T, C].
    Weight stored as [in, out, kernel] (PyTorch format); prepare_weights()
    transposes to [out, kernel, in] (MLX format).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, bias: bool = True):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.trim = kernel_size - stride
        scale = math.sqrt(2.0 / (in_channels * kernel_size))
        # Stored in PyTorch format; prepare_weights() transposes to MLX format
        self.weight = mx.random.normal((in_channels, out_channels, kernel_size)) * scale
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x):
        # x: [B, T, C]
        y = _conv_transpose1d(x, self.weight, self.bias, stride=self.stride)
        if self.trim > 0:
            y = y[:, :-self.trim, :]
        return y


# ============================================================
# ConvNeXt block (used in upsample path)
# ============================================================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with causal dwconv. Channels-last throughout."""
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.pwconv2 = nn.Linear(dim * 4, dim)
        self.gamma = mx.ones((dim,))

    def __call__(self, x):
        # x: [B, T, C] — no transposes needed
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        x = x * self.gamma
        return residual + x


# ============================================================
# Residual blocks
# ============================================================

class ResidualUnit(nn.Module):
    """Residual unit with causal dilated convolution."""
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.snake1 = Snake1d(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.snake2 = Snake1d(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def __call__(self, x):
        y = self.conv1(self.snake1(x))
        y = self.conv2(self.snake2(y))
        return x + y


class DecoderBlock(nn.Module):
    """Decoder upsampling block with causal convolutions."""
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake = Snake1d(input_dim)
        self.conv_t = CausalConvTranspose1d(input_dim, output_dim,
                                             kernel_size=stride * 2, stride=stride)
        self.res1 = ResidualUnit(output_dim, dilation=1)
        self.res2 = ResidualUnit(output_dim, dilation=3)
        self.res3 = ResidualUnit(output_dim, dilation=9)

    def __call__(self, x):
        x = self.snake(x)
        x = self.conv_t(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


# ============================================================
# RVQ Transformer (post-quantization) with RoPE + LayerScale
# ============================================================

class RVQTransformerBlock(nn.Module):
    """Transformer block with RoPE and LayerScale."""
    def __init__(self, dim: int, n_heads: int, intermediate: int):
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)

        head_dim = dim // n_heads
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # SwiGLU FFN
        self.w1 = nn.Linear(dim, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate, bias=False)

        # LayerScale
        self.attention_layer_scale = mx.ones((dim,))
        self.ffn_layer_scale = mx.ones((dim,))

    def __call__(self, x, freqs_cis, mask=None):
        B, T, D = x.shape

        h = self.attention_norm(x)
        qkv = self.wqkv(h)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = _apply_rotary_emb(q, freqs_cis[:T])
        k = _apply_rotary_emb(k, freqs_cis[:T])

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, D)
        attn = self.wo(attn)

        x = x + attn * self.attention_layer_scale

        h = self.ffn_norm(x)
        ffn_out = self.w2(nn.silu(self.w1(h)) * self.w3(h))
        x = x + ffn_out * self.ffn_layer_scale

        return x


def _apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Apply rotary embeddings.
    x: [B, n_heads, T, head_dim], freqs_cis: [T, head_dim/2, 2]
    """
    T = x.shape[2]
    head_dim = x.shape[3]
    x_pairs = x.reshape(*x.shape[:-1], head_dim // 2, 2)
    cos = freqs_cis[:T, :, 0][None, None, :, :]
    sin = freqs_cis[:T, :, 1][None, None, :, :]
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return mx.stack([out0, out1], axis=-1).reshape(x.shape)


class RVQPostModule(nn.Module):
    """RVQ post-quantization transformer with RoPE + LayerScale."""
    def __init__(self, config: DACConfig):
        super().__init__()
        dim = config.rvq_transformer_dim
        self.layers = [
            RVQTransformerBlock(dim, config.rvq_transformer_heads,
                                config.rvq_transformer_intermediate)
            for _ in range(config.rvq_transformer_layers)
        ]
        self.norm = nn.RMSNorm(dim)
        self.freqs_cis = mx.zeros((4096, config.rvq_transformer_dim // config.rvq_transformer_heads // 2, 2))

    def __call__(self, x):
        T = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(x.dtype)
        for layer in self.layers:
            x = layer(x, self.freqs_cis, mask=mask)
        return self.norm(x)


# ============================================================
# Upsample stage: CausalConvTranspose1d + ConvNeXt
# ============================================================

class UpsampleStage(nn.Module):
    """Upsample: CausalConvTranspose1d stride-2 + ConvNeXt block."""
    def __init__(self, dim: int, factor: int):
        super().__init__()
        self.conv = CausalConvTranspose1d(dim, dim, kernel_size=factor, stride=factor)
        self.cnx = ConvNeXtBlock(dim)

    def __call__(self, x):
        x = self.conv(x)
        x = self.cnx(x)
        return x


# ============================================================
# Codebook out_proj
# ============================================================

class CodebookOutProj(nn.Module):
    """Linear projection from codebook_dim to d_latent."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = mx.zeros((out_features, in_features))
        self.bias = mx.zeros((out_features,))

    def __call__(self, x):
        return x @ self.weight.T + self.bias


# ============================================================
# RVQ Decode Path
# ============================================================

class RVQDecode(nn.Module):
    """RVQ decode-only path: codes -> latent vectors."""
    def __init__(self, config: DACConfig):
        super().__init__()
        self.config = config
        self.semantic_codebook = nn.Embedding(config.codebook_size_semantic, config.codebook_dim)
        self.residual_codebooks = [
            nn.Embedding(config.codebook_size_residual, config.codebook_dim)
            for _ in range(config.n_residual_codebooks)
        ]
        self.semantic_out_proj = CodebookOutProj(config.codebook_dim, config.d_latent)
        self.residual_out_projs = [
            CodebookOutProj(config.codebook_dim, config.d_latent)
            for _ in range(config.n_residual_codebooks)
        ]
        self.post_module = RVQPostModule(config)
        self.upsample = [
            UpsampleStage(config.d_latent, factor)
            for factor in config.downsample_factor
        ]

    def __call__(self, codes: mx.array) -> mx.array:
        semantic = self.semantic_codebook(codes[:, 0, :])
        z = self.semantic_out_proj(semantic)
        for i in range(self.config.n_residual_codebooks):
            residual = self.residual_codebooks[i](codes[:, i + 1, :])
            z = z + self.residual_out_projs[i](residual)
        z = self.post_module(z)         # [B, T, d_latent] — already channels-last
        for stage in self.upsample:
            z = stage(z)
        return z


# ============================================================
# DAC Decoder
# ============================================================

class DACDecoder(nn.Module):
    """DAC decoder: latent -> audio waveform. Channels-last throughout."""
    def __init__(self, config: DACConfig):
        super().__init__()
        d = config.decoder_dim
        self.input_conv = CausalConv1d(config.d_latent, d, kernel_size=7)
        channels = [d]
        for _ in config.decoder_rates:
            channels.append(d // 2)
            d = d // 2
        self.blocks = []
        for i, stride in enumerate(config.decoder_rates):
            self.blocks.append(DecoderBlock(channels[i], channels[i + 1], stride))
        self.output_snake = Snake1d(channels[-1])
        self.output_conv = CausalConv1d(channels[-1], 1, kernel_size=7)

    def __call__(self, z: mx.array) -> mx.array:
        # z: [B, T, C] channels-last
        x = self.input_conv(z)
        for block in self.blocks:
            x = block(x)
        x = self.output_snake(x)
        x = self.output_conv(x)
        x = mx.tanh(x)
        return x


# ============================================================
# Weight preparation (PyTorch -> MLX format conversion)
# ============================================================

def _prepare_conv_weights(module):
    """Recursively transpose conv weights from PyTorch to MLX format.

    Conv1d: [out, in/groups, kernel] -> [out, kernel, in/groups]
    ConvTranspose1d: [in, out, kernel] -> [out, kernel, in]
    Snake1d alpha: [1, C, 1] -> [1, 1, C]

    Called once after load_weights().
    """
    if isinstance(module, CausalConv1d):
        if module.weight.ndim == 3:
            module.weight = module.weight.transpose(0, 2, 1)
    elif isinstance(module, CausalConvTranspose1d):
        if module.weight.ndim == 3:
            module.weight = module.weight.transpose(1, 2, 0)
    elif isinstance(module, Snake1d):
        if module.alpha.ndim == 3 and module.alpha.shape[2] == 1:
            module.alpha = module.alpha.transpose(0, 2, 1)

    # Recurse using MLX's children() which avoids circular refs
    children = module.children()
    for child in children.values() if isinstance(children, dict) else children:
        if isinstance(child, nn.Module):
            _prepare_conv_weights(child)
        elif isinstance(child, dict):
            for v in child.values():
                if isinstance(v, nn.Module):
                    _prepare_conv_weights(v)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, nn.Module):
                    _prepare_conv_weights(item)


# ============================================================
# Codec weight mapping (raw PyTorch → MLX model names)
# ============================================================

def _wn_weight(g: mx.array, v: mx.array) -> mx.array:
    """Compute weight-normalized weight: w = g * v / ||v||."""
    v_norm = mx.linalg.norm(v.reshape(v.shape[0], -1), axis=1, keepdims=True)
    # Reshape v_norm to broadcast with v
    for _ in range(v.ndim - 2):
        v_norm = mx.expand_dims(v_norm, axis=-1)
    return g * v / (v_norm + 1e-12)


def _map_transformer_layers(state, mapped, src_prefix, dst_prefix, n_layers):
    """Map transformer layer weights."""
    mapped[f"{dst_prefix}.freqs_cis"] = state[f"{src_prefix}.freqs_cis"]
    mapped[f"{dst_prefix}.norm.weight"] = state[f"{src_prefix}.norm.weight"]
    for i in range(n_layers):
        sp = f"{src_prefix}.layers.{i}"
        dp = f"{dst_prefix}.layers.{i}"
        mapped[f"{dp}.attention_norm.weight"] = state[f"{sp}.attention_norm.weight"]
        mapped[f"{dp}.wqkv.weight"] = state[f"{sp}.attention.wqkv.weight"]
        mapped[f"{dp}.wo.weight"] = state[f"{sp}.attention.wo.weight"]
        mapped[f"{dp}.attention_layer_scale"] = state[f"{sp}.attention_layer_scale.gamma"]
        mapped[f"{dp}.ffn_norm.weight"] = state[f"{sp}.ffn_norm.weight"]
        mapped[f"{dp}.w1.weight"] = state[f"{sp}.feed_forward.w1.weight"]
        mapped[f"{dp}.w2.weight"] = state[f"{sp}.feed_forward.w2.weight"]
        mapped[f"{dp}.w3.weight"] = state[f"{sp}.feed_forward.w3.weight"]
        mapped[f"{dp}.ffn_layer_scale"] = state[f"{sp}.ffn_layer_scale.gamma"]


def _map_convnext(state, mapped, src_prefix, dst_prefix):
    """Map a ConvNeXt block."""
    mapped[f"{dst_prefix}.dwconv.weight"] = state[f"{src_prefix}.dwconv.conv.weight"]
    mapped[f"{dst_prefix}.dwconv.bias"] = state[f"{src_prefix}.dwconv.conv.bias"]
    mapped[f"{dst_prefix}.norm.weight"] = state[f"{src_prefix}.norm.weight"]
    mapped[f"{dst_prefix}.norm.bias"] = state[f"{src_prefix}.norm.bias"]
    mapped[f"{dst_prefix}.pwconv1.weight"] = state[f"{src_prefix}.pwconv1.weight"]
    mapped[f"{dst_prefix}.pwconv1.bias"] = state[f"{src_prefix}.pwconv1.bias"]
    mapped[f"{dst_prefix}.pwconv2.weight"] = state[f"{src_prefix}.pwconv2.weight"]
    mapped[f"{dst_prefix}.pwconv2.bias"] = state[f"{src_prefix}.pwconv2.bias"]
    mapped[f"{dst_prefix}.gamma"] = state[f"{src_prefix}.gamma"]


def _map_residual_unit(state, mapped, src_prefix, dst_prefix):
    """Map a ResidualUnit (snake1 + dilated conv + snake2 + 1x1 conv)."""
    mapped[f"{dst_prefix}.snake1.alpha"] = state[f"{src_prefix}.0.alpha"]
    w = _wn_weight(state[f"{src_prefix}.1.weight_g"], state[f"{src_prefix}.1.weight_v"])
    mapped[f"{dst_prefix}.conv1.weight"] = w
    mapped[f"{dst_prefix}.conv1.bias"] = state[f"{src_prefix}.1.bias"]
    mapped[f"{dst_prefix}.snake2.alpha"] = state[f"{src_prefix}.2.alpha"]
    w = _wn_weight(state[f"{src_prefix}.3.weight_g"], state[f"{src_prefix}.3.weight_v"])
    mapped[f"{dst_prefix}.conv2.weight"] = w
    mapped[f"{dst_prefix}.conv2.bias"] = state[f"{src_prefix}.3.bias"]


def _map_raw_codec_decode(state):
    """Map raw PyTorch codec weights (mlx-community format) to MLX model names.

    Handles weight_g/weight_v pairs by pre-computing weight-normalized weights.
    """
    mapped = {}

    # 1. Semantic codebook + out_proj
    prefix = "quantizer.semantic_quantizer.quantizers.0"
    mapped["rvq.semantic_codebook.weight"] = state[f"{prefix}.codebook.weight"]
    w = _wn_weight(state[f"{prefix}.out_proj.weight_g"], state[f"{prefix}.out_proj.weight_v"])
    mapped["rvq.semantic_out_proj.weight"] = w.squeeze(-1)
    mapped["rvq.semantic_out_proj.bias"] = state[f"{prefix}.out_proj.bias"]

    # 2. Residual codebooks (9x) + out_projs
    for i in range(9):
        prefix = f"quantizer.quantizer.quantizers.{i}"
        mapped[f"rvq.residual_codebooks.{i}.weight"] = state[f"{prefix}.codebook.weight"]
        w = _wn_weight(state[f"{prefix}.out_proj.weight_g"], state[f"{prefix}.out_proj.weight_v"])
        mapped[f"rvq.residual_out_projs.{i}.weight"] = w.squeeze(-1)
        mapped[f"rvq.residual_out_projs.{i}.bias"] = state[f"{prefix}.out_proj.bias"]

    # 3. Post-module transformer (8 layers)
    _map_transformer_layers(state, mapped, "quantizer.post_module", "rvq.post_module", 8)

    # 4. Upsample (2 stages)
    for i in range(2):
        sp = f"quantizer.upsample.{i}"
        dp = f"rvq.upsample.{i}"
        mapped[f"{dp}.conv.weight"] = state[f"{sp}.0.conv.weight"]
        mapped[f"{dp}.conv.bias"] = state[f"{sp}.0.conv.bias"]
        _map_convnext(state, mapped, f"{sp}.1", f"{dp}.cnx")

    # 5. DAC Decoder
    w = _wn_weight(state["decoder.model.0.weight_g"], state["decoder.model.0.weight_v"])
    mapped["decoder.input_conv.weight"] = w
    mapped["decoder.input_conv.bias"] = state["decoder.model.0.bias"]

    for block_idx in range(4):
        sp = f"decoder.model.{block_idx + 1}"
        dp = f"decoder.blocks.{block_idx}"

        mapped[f"{dp}.snake.alpha"] = state[f"{sp}.block.0.alpha"]
        w = _wn_weight(state[f"{sp}.block.1.weight_g"], state[f"{sp}.block.1.weight_v"])
        mapped[f"{dp}.conv_t.weight"] = w
        mapped[f"{dp}.conv_t.bias"] = state[f"{sp}.block.1.bias"]

        for res_idx, res_name in enumerate(["res1", "res2", "res3"]):
            _map_residual_unit(state, mapped,
                               f"{sp}.block.{res_idx + 2}.block",
                               f"{dp}.{res_name}")

    mapped["decoder.output_snake.alpha"] = state["decoder.model.5.alpha"]
    w = _wn_weight(state["decoder.model.6.weight_g"], state["decoder.model.6.weight_v"])
    mapped["decoder.output_conv.weight"] = w
    mapped["decoder.output_conv.bias"] = state["decoder.model.6.bias"]

    return mapped


def load_codec_weights(codec, weights: dict):
    """Load codec weights, auto-detecting format (pre-converted or raw PyTorch).

    Args:
        codec: FishCodecDecoder instance
        weights: dict of weight name -> mx.array (from mx.load)
    """
    # Filter to decode-only keys
    dec_weights = {k: v for k, v in weights.items()
                   if not k.startswith("encoder.") and not k.startswith("rvq_enc.")}

    # Detect format: raw PyTorch has weight_g/weight_v pairs
    is_raw = any("weight_g" in k for k in dec_weights)

    if is_raw:
        mapped = _map_raw_codec_decode(dec_weights)
    else:
        mapped = dec_weights

    codec.load_weights(list(mapped.items()))
    codec.prepare_weights()


# ============================================================
# Full codec decode pipeline
# ============================================================

class FishCodecDecoder(nn.Module):
    """Complete decode pipeline: codes -> audio."""
    def __init__(self, config: DACConfig = DACConfig()):
        super().__init__()
        self.config = config
        self.rvq = RVQDecode(config)
        self.decoder = DACDecoder(config)

    def prepare_weights(self):
        """Transpose conv weights from PyTorch to MLX format. Call once after load_weights()."""
        _prepare_conv_weights(self)

    def __call__(self, codes: mx.array) -> mx.array:
        if codes.ndim == 2:
            codes = codes[None, :, :]
        z = self.rvq(codes)              # [B, T, d_latent] channels-last
        audio = self.decoder(z)          # [B, T, 1] channels-last
        return audio[:, :, 0]            # [B, T]
