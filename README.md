# Fish Audio S2 Pro — MLX

Optimized [MLX](https://github.com/ml-explore/mlx) inference for [Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro), a 4.5B parameter text-to-speech model running natively on Apple Silicon.

Model weights auto-download from [mlx-community/fish-audio-s2-pro-bf16](https://huggingface.co/mlx-community/fish-audio-s2-pro-bf16) on first run.

## Quick Start

```bash
# Generate speech (model downloads automatically)
uv run python run/e2e_pipeline.py --text "Hello, world!"

# With quantization for faster generation
uv run python run/e2e_pipeline.py --text "Hello, world!" --quantize int4

# Voice cloning from reference audio
uv run python run/e2e_pipeline.py \
    --text "Hello, world!" \
    --ref-audio voice.wav \
    --ref-text "transcript of the reference audio"

# Save voice imprint for reuse
uv run python run/e2e_pipeline.py \
    --text "Hello!" \
    --ref-audio voice.wav --ref-text "transcript" \
    --save-voice speaker.npz

# Reuse saved voice
uv run python run/e2e_pipeline.py --text "Hello again!" --voice speaker.npz
```

## Performance

Benchmarked subprocess-isolated on Apple Silicon (M4 Max, 64 GB):

| Config | RTF | ms/tok | Peak Memory | Notes |
|--------|-----|--------|-------------|-------|
| bf16 | 1.12x | 38.6 | 14.4 GB | Full precision, best quality baseline |
| slow-int8 | 1.33x | 32.2 | 11.0 GB | Only slow AR (4B) quantized to 8-bit |
| fast-int8 | 1.42x | 29.7 | 13.6 GB | Only fast AR (400M) quantized to 8-bit |
| slow-int4 | 1.48x | 28.7 | 9.0 GB | Only slow AR quantized to 4-bit |
| fast-int4 | 1.61x | 26.0 | 13.6 GB | Only fast AR quantized to 4-bit |
| int8 | 1.80x | 23.1 | 10.5 GB | Both ARs 8-bit |
| s8+f4 | 2.11x | 19.4 | 10.4 GB | Slow 8-bit, fast 4-bit |
| s4+f8 | 2.09x | 19.5 | 8.6 GB | Slow 4-bit, fast 8-bit. Same speed, least memory |
| int4 | 2.55x | 15.6 | 8.7 GB | Both ARs 4-bit. Fastest |

RTF = real-time factor (audio duration / compute time). Values >1.0 are faster than real-time.

All configs generate above real-time. int4 produces audio 2.55x faster than playback speed.

## Features

- **Voice cloning** — single speaker from reference audio, or multi-speaker with `<|speaker:N|>` tags
- **Long-form synthesis** — automatic text chunking with cross-fade stitching
- **Voice caching** — pre-fill KV cache from voice reference, reuse across utterances
- **Quantization** — independent slow/fast AR quantization (bf16, int8, int4, or mixed)
- **Speed control** — adjustable playback speed via `--speed`

## Architecture

```
Text ─→ [Slow AR: 36-layer Qwen3, 4B] ─→ semantic tokens
                                              │
                                              ▼
         [Fast AR: 4-layer decoder, 400M] ─→ 10 codebooks/frame
                                              │
                                              ▼
         [DAC Codec Decoder] ─→ 44.1kHz audio waveform
```

## Optimizations

### KV Cache

The Slow AR generates tokens autoregressively through 36 transformer layers. A naive implementation concatenates new K/V tensors to the cache at every step — 72 growing-array allocations per token, O(n²) total memory traffic. We pre-allocate fixed buffers and write into them with offset tracking, the same approach used by the Fast AR's rotating cache. The buffers double when capacity is exceeded.

### Channels-Last Codec

The DAC codec decoder uses ~20 causal convolution layers. MLX's `conv1d` operates on channels-last `[B, T, C]` tensors natively, but the original PyTorch implementation stores data as `[B, C, T]`, requiring 3 transposes per convolution call (~60 total). We store all data channels-last and transpose the convolution weights once at load time via `prepare_weights()`, eliminating all runtime transposes.

### Fused Snake1d Metal Kernel

The Snake activation `x + sin²(αx) / α` appears ~25 times in the codec decoder. The decomposed version requires 4 elementwise ops + 1 sin, each reading/writing the full tensor. A custom Metal kernel via `mx.fast.metal_kernel` fuses this into a single pass — one read, one sin, one write.

### Projected Output Weights

The Slow AR vocabulary is 155,776 tokens, but during generation only ~4,097 are valid (4,096 semantic tokens + EOS). Instead of computing the full `[dim, 155K]` output projection and discarding 97% of it, we pre-extract the valid rows from the embedding matrix and compute a `[dim, 4097]` matmul — 38x smaller per step.

### Speculative Fast AR

Each generation step runs both the Slow AR (predict semantic token) and Fast AR (fill 9 residual codebooks). A naive implementation would `eval()` the Slow AR result, check for EOS, then run the Fast AR — two GPU pipeline stalls per step. Instead, we run the Fast AR speculatively before evaluating the Slow AR token. If it turns out to be EOS, we discard one wasted Fast AR pass. On every non-EOS step (the vast majority), we eliminate a GPU stall.

### Batched Fast AR Prime

The Fast AR generates 10 codebook tokens per frame. The first two operations — priming the KV cache with the Slow AR hidden state and predicting the first codebook — were originally two separate forward passes through all 4 layers. By concatenating them into a single L=2 input with a causal mask, we eliminate one full forward pass per generation step (~5% of Fast AR cost).

### Vectorized Repetition Penalty

The repetition penalty needs to track recently generated token IDs and penalize their logits. A Python list with per-step iteration is replaced by a pre-allocated `mx.array` ring buffer with pure MLX indexing — no Python loops during generation.

### Pre-allocated Output Buffer

Generated codebook columns are written into a pre-allocated `[11, max_tokens]` buffer via index assignment, avoiding per-step `mx.array` creation, `mx.concatenate`, list appends, and the final `mx.stack`.

## CLI Options

```
--text TEXT            Text to synthesize (required)
--model MODEL         HF model ID or local path (default: mlx-community/fish-audio-s2-pro-bf16)
--output PATH         Output WAV path (default: output.wav)
--quantize int4|int8  Quantize full model
--quantize-slow ...   Quantize slow AR only
--quantize-fast ...   Quantize fast AR only
--ref-audio PATH      Reference audio for voice cloning
--ref-text TEXT        Transcript of reference audio
--voice PATH [PATH]   Pre-encoded voice imprint(s) (.npz)
--save-voice PATH     Save encoded voice imprint for reuse
--cache-voice         Cache voice KV state for faster prefill
--temperature FLOAT   Sampling temperature (default: 0.7)
--top-p FLOAT         Top-p sampling (default: 0.8)
--max-tokens INT      Max generation tokens (default: 2048)
--speed FLOAT         Playback speed multiplier (default: 1.0)
--chunk-size INT      Max chars per chunk for long-form (default: 200)
--seed INT            Random seed (default: 42)
```

## Project Structure

```
run/
  e2e_pipeline.py    CLI entry point
  model.py           Dual-AR model (SlowAR + FastAR)
  generate.py        Generation pipeline, sampling, text chunking
  load_weights.py    HuggingFace weight loading
  dac_decoder.py     DAC codec decoder (channels-last, fused Snake1d)
  dac_encoder.py     DAC codec encoder (for voice reference encoding)
bench_compare.py     Quantization sweep benchmark
```

## Requirements

- Python >= 3.10
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX >= 0.22.0

```bash
uv sync
```

## License

This inference code is MIT licensed. See [LICENSE](LICENSE).

The model weights ([fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)) are under the [Fish Audio Research License](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) — free for research and non-commercial use, commercial use requires a separate license from Fish Audio.
