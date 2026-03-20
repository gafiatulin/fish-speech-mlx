#!/usr/bin/env python3
"""Benchmark: Fish Audio S2 Pro MLX quantization sweep.

Each config runs in a separate subprocess for clean GPU state.
Measures ms/tok, RTF, peak GPU memory, and codec decode time.

Run from fish-speech directory:
    uv run python bench_compare.py --voice speaker.npz
    uv run python bench_compare.py --voice speaker.npz --text "Custom text"
"""

import argparse
import json
import subprocess
import sys
import textwrap

DEFAULT_TEXT = "Hello, this is a test of the optimized sampling pipeline. The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon."


def make_script(text, voice, seed, quantize=None, quantize_slow=None, quantize_fast=None):
    q_arg = f'"{quantize}"' if quantize else "None"
    qs_arg = f'"{quantize_slow}"' if quantize_slow else "None"
    qf_arg = f'"{quantize_fast}"' if quantize_fast else "None"
    voice_repr = repr(voice) if voice else "None"
    return textwrap.dedent(f"""\
import json, time, sys, numpy as np, mlx.core as mx
from transformers import AutoTokenizer
from fish_speech_mlx.load_weights import load_model, resolve_model_path
from fish_speech_mlx.generate import build_prompt, build_prompt_with_reference, generate, GenerationConfig
from fish_speech_mlx.dac_decoder import FishCodecDecoder, load_codec_weights

model_path = resolve_model_path("mlx-community/fish-audio-s2-pro-bf16")
tokenizer = AutoTokenizer.from_pretrained(str(model_path))
model = load_model(model_path, quantize={q_arg},
                   quantize_slow={qs_arg}, quantize_fast={qf_arg})

codec = FishCodecDecoder()
load_codec_weights(codec, mx.load(str(model_path / "codec.safetensors")))

voice_path = {voice_repr}
if voice_path is not None:
    voice = np.load(voice_path, allow_pickle=False)
    ref_codes = mx.array(voice["codes"])
    ref_text = str(voice["text"])

config = GenerationConfig(temperature=0.7, top_p=0.8)

mx.reset_peak_memory()
mx.random.seed({seed})

if voice_path is not None:
    prompt = build_prompt_with_reference({text!r}, ref_codes, ref_text, tokenizer)
else:
    prompt = build_prompt({text!r}, tokenizer)
text_token_count = len(tokenizer.encode({text!r}, add_special_tokens=False))

t0 = time.time()
codes, gen_metrics = generate(model, prompt, config, text_token_count=text_token_count)
mx.eval(codes)
gen_s = time.time() - t0

t1 = time.time()
audio = codec(codes)
mx.eval(audio)
dec_s = time.time() - t1

peak = mx.get_peak_memory() / 1e9
audio_np = np.array(audio[0])
duration = len(audio_np) / 44100
total_s = gen_s + dec_s

print(json.dumps({{
    "total_s": total_s,
    "gen_s": gen_s,
    "dec_s": dec_s,
    "duration_s": duration,
    "rtf": duration / total_s if total_s > 0 else 0,
    "peak_mem_gb": peak,
    "ms_tok": gen_metrics["total_ms"] / max(gen_metrics["n_tokens"], 1),
    "n_tokens": gen_metrics["n_tokens"],
}}))
""")


def run_bench(script):
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    stderr = (result.stdout + result.stderr)[-500:]
    print(f"\n  FAILED: {stderr}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Fish Audio S2 Pro quantization sweep benchmark")
    parser.add_argument("--voice", default=None, help="Path to voice imprint (.npz). Omit for default voice.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    args = parser.parse_args()

    configs = [
        ("bf16",       {}),
        ("slow-int8",  {"quantize_slow": "int8"}),
        ("fast-int8",  {"quantize_fast": "int8"}),
        ("slow-int4",  {"quantize_slow": "int4"}),
        ("fast-int4",  {"quantize_fast": "int4"}),
        ("int8",       {"quantize": "int8"}),
        ("s8+f4",      {"quantize_slow": "int8", "quantize_fast": "int4"}),
        ("s4+f8",      {"quantize_slow": "int4", "quantize_fast": "int8"}),
        ("int4",       {"quantize": "int4"}),
    ]

    print("=" * 82)
    print("BENCHMARK: Fish Audio S2 Pro — quantization sweep (subprocess isolation)")
    print(f"Text: {args.text[:60]}...")
    print(f"Voice: {args.voice}")
    print("=" * 82)

    all_results = []

    for name, kwargs in configs:
        print(f"Running {name}...", end=" ", flush=True)
        script = make_script(args.text, args.voice, args.seed, **kwargs)
        r = run_bench(script)
        if r:
            all_results.append((name, r))
            print(f"RTF={r['rtf']:.2f}x  mem={r['peak_mem_gb']:.1f}GB  "
                  f"ms/tok={r['ms_tok']:.1f}  decode={r['dec_s']:.2f}s")
        else:
            print("FAILED")

    # Summary
    print("\n" + "=" * 82)
    print(f"{'Config':<22} {'RTF':>7} {'ms/tok':>8} {'Peak Mem':>10} {'Decode':>8} {'Total':>8}")
    print("-" * 82)
    for name, r in all_results:
        print(f"{name:<22} {r['rtf']:>6.2f}x {r['ms_tok']:>7.1f} {r['peak_mem_gb']:>8.1f}GB "
              f"{r['dec_s']:>7.2f}s {r['total_s']:>7.2f}s")
    print("=" * 82)


if __name__ == "__main__":
    main()
