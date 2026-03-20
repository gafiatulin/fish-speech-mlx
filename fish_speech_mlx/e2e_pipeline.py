#!/usr/bin/env python3
"""Fish Audio S2 Pro — MLX end-to-end TTS pipeline.

Usage:
    fish-speech-mlx --text "Hello, world!"
    fish-speech-mlx --text "Hello!" --quantize int4
    fish-speech-mlx --text "Hello!" --ref-audio voice.wav --ref-text "transcript"
    fish-speech-mlx --text "Hello!" --voice speaker1.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf


def encode_reference(audio_path: Path, transcript: str,
                     model_path: Path) -> dict:
    """Encode reference audio into a voice imprint.

    Returns dict with 'codes' (int32 ndarray [10, T]) and 'text' (str).
    """
    audio_data, sr = sf.read(str(audio_path))
    if sr != 44100:
        import scipy.signal
        audio_data = scipy.signal.resample_poly(audio_data, 44100, sr)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    audio_mx = mx.array(audio_data.astype(np.float32))[None, None, :]

    from .dac_encoder import FishCodecEncoder, DACConfig, load_codec_encoder_weights

    codec_path = model_path / "codec.safetensors"
    if not codec_path.exists():
        raise FileNotFoundError(f"Codec not found: {codec_path}")

    codec_enc = FishCodecEncoder(DACConfig())
    load_codec_encoder_weights(codec_enc, mx.load(str(codec_path)))

    t0 = time.time()
    codes = codec_enc(audio_mx)
    mx.eval(codes)
    elapsed = (time.time() - t0) * 1000
    codes_np = np.array(codes[0])  # [10, T]

    print(f"  Encoded {audio_path.name} in {elapsed:.0f}ms -> {codes_np.shape[1]} frames")
    return {"codes": codes_np, "text": transcript}


def save_voice(imprint: dict, path: Path):
    """Save voice imprint to .npz file."""
    np.savez(str(path), codes=imprint["codes"],
             text=np.array(imprint["text"]))
    print(f"  Saved voice imprint: {path}")


def load_voice(path: Path) -> dict:
    """Load voice imprint from .npz file."""
    data = np.load(str(path), allow_pickle=False)
    return {"codes": data["codes"], "text": str(data["text"])}


def main():
    from .load_weights import DEFAULT_MODEL

    parser = argparse.ArgumentParser(description="Fish Audio S2 Pro MLX TTS")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"HF model ID or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", type=Path, default=Path("output.wav"),
                        help="Output WAV path")
    parser.add_argument("--quantize", choices=["int4", "int8"], default=None,
                        help="Full model quantization")
    parser.add_argument("--quantize-fast", choices=["int4", "int8"], default=None,
                        help="Fast AR only quantization")
    parser.add_argument("--quantize-slow", choices=["int4", "int8"], default=None,
                        help="Slow AR only quantization")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (>1.0 faster, <1.0 slower)")
    parser.add_argument("--ref-audio", type=Path, default=None,
                        help="Reference audio for voice cloning")
    parser.add_argument("--ref-text", type=str, default=None,
                        help="Transcript of reference audio")
    parser.add_argument("--voice", type=Path, nargs="+", default=None,
                        help="Pre-encoded voice imprint(s) (.npz). Multiple for multi-speaker.")
    parser.add_argument("--save-voice", type=Path, default=None,
                        help="Save encoded voice imprint to .npz for reuse")
    parser.add_argument("--cache-voice", action="store_true",
                        help="Cache voice KV state for faster prefill")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Max characters per chunk for long-form generation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mx.random.seed(args.seed)

    print("=" * 60)
    print("Fish Audio S2 Pro — MLX Pipeline")
    print("=" * 60)
    print(f"Text: {args.text[:80]}...")
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantize or 'none'}")
    print()

    # 1. Resolve model path and load
    from .load_weights import load_model, resolve_model_path
    model_path = resolve_model_path(args.model)

    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    print("Loading LM...")
    t0 = time.time()
    model = load_model(model_path, quantize=args.quantize,
                       quantize_fast=args.quantize_fast,
                       quantize_slow=args.quantize_slow)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # 2. Build prompt
    from .generate import (
        build_prompt, build_prompt_with_reference, build_prompt_multi_speaker,
        build_voice_prefix, build_text_suffix, prefill_voice,
        generate, GenerationConfig, chunk_text,
    )

    ref_codes = None
    ref_text = None
    speakers = None

    if args.voice is not None:
        if len(args.voice) == 1:
            print(f"Loading voice imprint: {args.voice[0]}")
            imprint = load_voice(args.voice[0])
            ref_codes = mx.array(imprint["codes"])
            ref_text = imprint["text"]
            print(f"  Loaded: {ref_codes.shape[1]} frames, transcript: {ref_text[:60]}...")
        else:
            speakers = []
            for vpath in args.voice:
                print(f"Loading voice imprint: {vpath}")
                imprint = load_voice(vpath)
                codes = mx.array(imprint["codes"])
                text = imprint["text"]
                speakers.append({"codes": codes, "text": text})
                print(f"  Speaker {len(speakers)-1}: {codes.shape[1]} frames, transcript: {text[:60]}...")
            print(f"  {len(speakers)} speakers loaded")

    elif args.ref_audio is not None:
        if args.ref_text is None:
            raise ValueError("--ref-text required when using --ref-audio")

        print("Encoding reference audio...")
        imprint = encode_reference(args.ref_audio, args.ref_text, model_path)
        ref_codes = mx.array(imprint["codes"])
        ref_text = args.ref_text

        if args.save_voice is not None:
            save_voice(imprint, args.save_voice)

    config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Split text into chunks for long-form generation
    chunks = chunk_text(args.text, max_chars=args.chunk_size)
    use_chunking = len(chunks) > 1

    # Voice cache: use for chunking or when explicitly requested
    voice_cache = None
    if ref_codes is not None and speakers is None and (args.cache_voice or use_chunking):
        print("Prefilling voice cache...")
        t_vc = time.time()
        voice_prefix = build_voice_prefix(ref_codes, ref_text, tokenizer)
        voice_cache = prefill_voice(model, voice_prefix)
        print(f"  Voice cache ready: {voice_prefix.shape[1]} tokens in {(time.time()-t_vc)*1000:.0f}ms")

    if use_chunking:
        print(f"\nLong-form: {len(chunks)} chunks")

    # 3. Generate codes (per chunk)
    all_codes = []
    total_metrics = {"prefill_ms": 0, "step_ms": 0,
                     "total_ms": 0, "n_tokens": 0}

    for i, chunk in enumerate(chunks):
        if use_chunking:
            print(f"\n--- Chunk {i+1}/{len(chunks)}: {chunk[:50]}{'...' if len(chunk)>50 else ''}")

        if voice_cache is not None:
            prompt = build_text_suffix(chunk, tokenizer)
        elif speakers is not None:
            prompt = build_prompt_multi_speaker(chunk, speakers, tokenizer)
        elif ref_codes is not None:
            prompt = build_prompt_with_reference(
                chunk, ref_codes, ref_text, tokenizer
            )
        else:
            prompt = build_prompt(chunk, tokenizer)

        if not use_chunking:
            print(f"Prompt shape: {prompt.shape} (11 x {prompt.shape[1]} tokens)")

        print(f"{'  ' if use_chunking else ''}Generating{'...' if not use_chunking else ''}", end=" ", flush=True)
        text_token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
        codes, metrics = generate(model, prompt, config, voice_cache=voice_cache,
                                  text_token_count=text_token_count)
        n = metrics["n_tokens"]
        print(f"{n} tokens in {metrics['total_ms']:.0f}ms ({metrics['total_ms']/max(n,1):.1f}ms/tok)")

        if n > 0:
            all_codes.append(codes)

            if i == 0 and use_chunking and voice_cache is None and speakers is None:
                ref_text = chunk
                ref_codes = codes
                print("  Building voice cache from first chunk...")
                t_vc = time.time()
                voice_prefix = build_voice_prefix(ref_codes, ref_text, tokenizer)
                voice_cache = prefill_voice(model, voice_prefix)
                print(f"  Voice cache ready: {voice_prefix.shape[1]} tokens in {(time.time()-t_vc)*1000:.0f}ms")

        for k in total_metrics:
            total_metrics[k] += metrics[k]

    n = total_metrics["n_tokens"]
    total = total_metrics["total_ms"]

    print(f"\nGeneration complete:")
    print(f"  Tokens: {n}")
    print(f"  Prefill: {total_metrics['prefill_ms']:.0f}ms")
    print(f"  Step: {total_metrics['step_ms']:.0f}ms ({total_metrics['step_ms']/max(n,1):.1f}ms/tok)")
    print(f"  Total: {total:.0f}ms ({total/max(n,1):.1f}ms/tok)")

    if not all_codes:
        print("No tokens generated!")
        return

    # 4. Decode codes to audio
    print("\nDecoding audio...")

    codec_path = model_path / "codec.safetensors"
    if not codec_path.exists():
        codes = mx.concatenate(all_codes, axis=1) if len(all_codes) > 1 else all_codes[0]
        print(f"  No codec found at {codec_path}")
        print(f"  Saving raw codes to {args.output.with_suffix('.npy')}")
        np.save(str(args.output.with_suffix(".npy")), np.array(codes))
        return

    t_dec = time.time()

    from .dac_decoder import FishCodecDecoder, load_codec_weights
    codec = FishCodecDecoder()
    load_codec_weights(codec, mx.load(str(codec_path)))

    print(f"  Using MLX codec: {codec_path}")

    def _decode_codes(codes):
        audio = codec(codes)
        mx.eval(audio)
        return np.array(audio[0])

    if len(all_codes) == 1:
        audio_np = _decode_codes(all_codes[0])
    else:
        crossfade_samples = int(0.05 * 44100)  # 50ms cross-fade
        audio_chunks = [_decode_codes(c) for c in all_codes]

        audio_np = audio_chunks[0]
        for chunk_audio in audio_chunks[1:]:
            fade = min(crossfade_samples, len(audio_np), len(chunk_audio))
            if fade > 0:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                audio_np[-fade:] = audio_np[-fade:] * (1.0 - ramp) + chunk_audio[:fade] * ramp
                audio_np = np.concatenate([audio_np, chunk_audio[fade:]])
            else:
                audio_np = np.concatenate([audio_np, chunk_audio])

    decode_ms = (time.time() - t_dec) * 1000
    print(f"  Codec decode: {decode_ms:.0f}ms")

    # Speed adjustment
    if args.speed != 1.0:
        from .generate import adjust_speed
        audio_mx = mx.array(audio_np)
        audio_mx = adjust_speed(audio_mx, args.speed)
        mx.eval(audio_mx)
        audio_np = np.array(audio_mx)
        print(f"  Speed: {args.speed:.2f}x")

    duration = len(audio_np) / 44100

    # 5. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio_np, 44100)
    print(f"\nSaved: {args.output}")
    print(f"Duration: {duration:.1f}s")

    total_with_decode = total + decode_ms
    print(f"RTF: {duration / (total_with_decode / 1000):.2f}x (audio_time/compute_time)")


if __name__ == "__main__":
    main()
