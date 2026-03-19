"""Fish Audio S2 Pro — MLX generation pipeline.

Implements the full text-to-audio pipeline:
  1. Tokenize text into prompt
  2. Slow AR generates semantic tokens autoregressively
  3. Fast AR fills residual codebooks per frame
  4. DAC decoder converts codes to audio
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from model import DualARModel, KVCache, RotatingKVCache


# Special token IDs
IM_START = 151644
IM_END = 151645
PAD_TOKEN = 151669
TEXT_TOKEN = 151672
VOICE_TOKEN = 151673
AUDIO_START = 151675
AUDIO_END = 151676
AUDIO_PAD = 151677
SEMANTIC_START = 151678
SEMANTIC_END = 155773


@dataclass
class GenerationConfig:
    """Generation hyperparameters."""
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 0
    repetition_penalty: float = 1.1
    repetition_window: int = 16


def _sample_token(logits: mx.array, temperature: float = 1.0,
                  top_p: float = 1.0) -> mx.array:
    """Sample a token from logits with temperature and top-p."""
    if logits.ndim == 1:
        logits = logits[None, :]

    if temperature <= 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    if top_p < 1.0:
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        probs_sorted = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(probs_sorted, axis=-1)
        mask = (cumulative_probs - probs_sorted) >= top_p
        sorted_logits = mx.where(mask, -1e9, sorted_logits)
        unsort_indices = mx.argsort(sorted_indices, axis=-1)
        logits = mx.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    return mx.random.categorical(logits)


def _sample_simple(logits: mx.array, temperature: float = 1.0) -> mx.array:
    """Fast sampling with temperature only (no top-p). Used for codebook tokens."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits / temperature)


def build_prompt(text: str, tokenizer) -> mx.array:
    """Build the token prompt for generation.

    Returns tensor of shape [11, T] where:
      - Row 0: vocab token IDs
      - Rows 1-10: codebook indices (all AUDIO_PAD for text positions)
    """
    tokens = []

    # System turn
    tokens.append(IM_START)
    tokens.extend(tokenizer.encode("system\n", add_special_tokens=False))
    tokens.extend(tokenizer.encode("convert the provided text to speech", add_special_tokens=False))
    tokens.append(IM_END)
    tokens.append(tokenizer.encode("\n", add_special_tokens=False)[0])

    # User turn
    tokens.append(IM_START)
    tokens.extend(tokenizer.encode("user\n", add_special_tokens=False))
    tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    tokens.append(IM_END)
    tokens.append(tokenizer.encode("\n", add_special_tokens=False)[0])

    # Assistant turn
    tokens.append(IM_START)
    tokens.extend(tokenizer.encode("assistant\n", add_special_tokens=False))
    tokens.append(VOICE_TOKEN)

    T = len(tokens)
    prompt = mx.full((11, T), AUDIO_PAD, dtype=mx.int32)
    prompt[0, :] = mx.array(tokens, dtype=mx.int32)

    return prompt


def build_prompt_with_reference(text: str, ref_codes: mx.array,
                                ref_text: str, tokenizer) -> mx.array:
    """Build prompt with voice reference for cloning."""
    tokens_part1 = []

    # System turn with reference
    tokens_part1.append(IM_START)
    tokens_part1.extend(tokenizer.encode("system\n", add_special_tokens=False))
    sys_text = "convert the provided text to speech reference to the following:\n\nText:\n"
    tokens_part1.extend(tokenizer.encode(sys_text + ref_text, add_special_tokens=False))
    tokens_part1.extend(tokenizer.encode("\n\nSpeech:\n", add_special_tokens=False))

    T1 = len(tokens_part1)
    T_ref = ref_codes.shape[1]
    ref_semantic = ref_codes[0, :] + SEMANTIC_START

    tokens_part2 = []
    tokens_part2.append(IM_END)
    tokens_part2.append(tokenizer.encode("\n", add_special_tokens=False)[0])

    tokens_part2.append(IM_START)
    tokens_part2.extend(tokenizer.encode("user\n", add_special_tokens=False))
    tokens_part2.append(TEXT_TOKEN)
    tokens_part2.extend(tokenizer.encode(text, add_special_tokens=False))
    tokens_part2.append(IM_END)
    tokens_part2.append(tokenizer.encode("\n", add_special_tokens=False)[0])

    tokens_part2.append(IM_START)
    tokens_part2.extend(tokenizer.encode("assistant\n", add_special_tokens=False))
    tokens_part2.append(VOICE_TOKEN)

    T2 = len(tokens_part2)
    T_total = T1 + T_ref + T2

    prompt = mx.full((11, T_total), AUDIO_PAD, dtype=mx.int32)
    prompt[0, :T1] = mx.array(tokens_part1, dtype=mx.int32)
    prompt[0, T1:T1 + T_ref] = ref_semantic
    for i in range(10):
        prompt[i + 1, T1:T1 + T_ref] = ref_codes[i, :]
    prompt[0, T1 + T_ref:] = mx.array(tokens_part2, dtype=mx.int32)

    return prompt


def build_prompt_multi_speaker(text: str, speakers: list[dict],
                               tokenizer) -> mx.array:
    """Build prompt with multiple voice references.

    Args:
        text: Text with <|speaker:N|> tags to control voice switching.
        speakers: List of dicts with 'codes' (mx.array [10, T]) and 'text' (str).
            Speaker index matches list position.
    """
    tokens_part1 = []
    tokens_part1.append(IM_START)
    tokens_part1.extend(tokenizer.encode("system\n", add_special_tokens=False))
    sys_text = "convert the provided text to speech reference to the following:\n\nText:\n"

    # Build tagged transcript for all speakers
    tagged_texts = []
    for idx, spk in enumerate(speakers):
        tag = f"<|speaker:{idx}|>"
        tagged_texts.append(f"{tag}{spk['text']}")
    sys_text += "\n".join(tagged_texts)

    tokens_part1.extend(tokenizer.encode(sys_text, add_special_tokens=False))
    tokens_part1.extend(tokenizer.encode("\n\nSpeech:\n", add_special_tokens=False))

    T1 = len(tokens_part1)

    # Concatenate all speaker codes
    all_codes = mx.concatenate([spk["codes"] for spk in speakers], axis=1)
    T_ref = all_codes.shape[1]
    ref_semantic = all_codes[0, :] + SEMANTIC_START

    tokens_part2 = []
    tokens_part2.append(IM_END)
    tokens_part2.append(tokenizer.encode("\n", add_special_tokens=False)[0])

    tokens_part2.append(IM_START)
    tokens_part2.extend(tokenizer.encode("user\n", add_special_tokens=False))
    tokens_part2.append(TEXT_TOKEN)
    tokens_part2.extend(tokenizer.encode(text, add_special_tokens=False))
    tokens_part2.append(IM_END)
    tokens_part2.append(tokenizer.encode("\n", add_special_tokens=False)[0])

    tokens_part2.append(IM_START)
    tokens_part2.extend(tokenizer.encode("assistant\n", add_special_tokens=False))
    tokens_part2.append(VOICE_TOKEN)

    T2 = len(tokens_part2)
    T_total = T1 + T_ref + T2

    prompt = mx.full((11, T_total), AUDIO_PAD, dtype=mx.int32)
    prompt[0, :T1] = mx.array(tokens_part1, dtype=mx.int32)
    prompt[0, T1:T1 + T_ref] = ref_semantic
    for i in range(10):
        prompt[i + 1, T1:T1 + T_ref] = all_codes[i, :]
    prompt[0, T1 + T_ref:] = mx.array(tokens_part2, dtype=mx.int32)

    return prompt


def build_voice_prefix(ref_codes: mx.array, ref_text: str,
                       tokenizer) -> mx.array:
    """Build the voice-only prefix prompt (cacheable across generations).

    Returns tensor of shape [11, T_prefix] containing the system turn
    with voice reference, ending at IM_END + newline.
    """
    tokens = []
    tokens.append(IM_START)
    tokens.extend(tokenizer.encode("system\n", add_special_tokens=False))
    sys_text = "convert the provided text to speech reference to the following:\n\nText:\n"
    tokens.extend(tokenizer.encode(sys_text + ref_text, add_special_tokens=False))
    tokens.extend(tokenizer.encode("\n\nSpeech:\n", add_special_tokens=False))

    T_text = len(tokens)
    T_ref = ref_codes.shape[1]
    ref_semantic = ref_codes[0, :] + SEMANTIC_START

    # End of system turn
    end_tokens = []
    end_tokens.append(IM_END)
    end_tokens.append(tokenizer.encode("\n", add_special_tokens=False)[0])
    T_end = len(end_tokens)

    T_total = T_text + T_ref + T_end
    prefix = mx.full((11, T_total), AUDIO_PAD, dtype=mx.int32)
    prefix[0, :T_text] = mx.array(tokens, dtype=mx.int32)
    prefix[0, T_text:T_text + T_ref] = ref_semantic
    for i in range(10):
        prefix[i + 1, T_text:T_text + T_ref] = ref_codes[i, :]
    prefix[0, T_text + T_ref:] = mx.array(end_tokens, dtype=mx.int32)

    return prefix


def build_text_suffix(text: str, tokenizer) -> mx.array:
    """Build the text-only suffix prompt (changes per generation).

    Returns tensor of shape [11, T_suffix] for the user turn + assistant start.
    """
    tokens = []
    tokens.append(IM_START)
    tokens.extend(tokenizer.encode("user\n", add_special_tokens=False))
    tokens.append(TEXT_TOKEN)
    tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    tokens.append(IM_END)
    tokens.append(tokenizer.encode("\n", add_special_tokens=False)[0])
    tokens.append(IM_START)
    tokens.extend(tokenizer.encode("assistant\n", add_special_tokens=False))
    tokens.append(VOICE_TOKEN)

    T = len(tokens)
    suffix = mx.full((11, T), AUDIO_PAD, dtype=mx.int32)
    suffix[0, :] = mx.array(tokens, dtype=mx.int32)
    return suffix


def prefill_voice(model: DualARModel, voice_prefix: mx.array
                  ) -> list:
    """Prefill the voice prefix and return cached KV state.

    Returns list of (keys, values) tuples for each slow AR layer.
    """
    slow_cfg = model.slow_config

    prompt_tokens = voice_prefix[0:1, :]
    prompt_codebooks = voice_prefix[1:, :].transpose(1, 0)[None, :, :]
    T = voice_prefix.shape[1]

    slow_cache = [
        KVCache(slow_cfg.n_kv_heads, slow_cfg.head_dim, max_len=T)
        for _ in range(slow_cfg.n_layers)
    ]

    mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
    mask = mask.astype(mx.bfloat16)

    x = model.slow.embed(prompt_tokens, prompt_codebooks)
    model.slow(x, mask=mask, cache=slow_cache)

    # Materialize and return cache state (sliced to actual length)
    state = []
    for c in slow_cache:
        k = c.keys[:, :, :c.offset, :]
        v = c.values[:, :, :c.offset, :]
        mx.eval(k, v)
        state.append((k, v))
    return state


def _fast_ar_generate(model, last_hidden: mx.array, code_0: mx.array,
                      fast_cache: list, temperature: float) -> mx.array:
    """Generate all 10 codebook codes for one frame.

    Uses simple temperature sampling (no top-p) for speed.
    Keeps everything as mx.arrays — no .item() calls, single eval at end.

    Batches the prime (hidden state) + first codebook into a single forward
    pass with L=2, saving one full Fast AR forward pass (~10% Fast AR cost).

    Returns:
        codes: [10] codebook indices
    """
    codes = [code_0]

    # Batch prime + first codebook: single forward pass instead of two.
    # Position 0 = slow AR hidden state (prime), position 1 = code_0 embedding.
    # Causal mask ensures position 0 only attends to itself.
    cb0_embed = model.fast.embeddings(code_0.reshape(1, 1))  # [1, 1, dim]
    combined = mx.concatenate([last_hidden, cb0_embed], axis=1)  # [1, 2, dim]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(2)
    mask = mask.astype(combined.dtype)
    fast_logits = model.fast(combined, mask=mask, cache=fast_cache)
    cb_token = _sample_simple(fast_logits[0, -1, :], temperature=temperature)
    codes.append(cb_token)

    # Steps 2-9
    for _ in range(2, model.fast_config.num_codebooks):
        cb_input = model.fast.embeddings(codes[-1].reshape(1, 1))
        fast_logits = model.fast(cb_input, cache=fast_cache)
        cb_token = _sample_simple(fast_logits[0, -1, :], temperature=temperature)
        codes.append(cb_token)

    return mx.stack(codes)  # [10]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving abbreviations.

    Handles common abbreviations (Dr., Mr., Mrs., Ms., Prof., St., Jr.,
    Sr., vs., etc., e.g., i.e.), acronyms (U.S., M.I.T., E.U.), and
    decimal numbers (3.5%).
    """
    import re

    # Find sentence boundaries: positions after .!? + whitespace where
    # the next char is uppercase and the word before the punctuation
    # is not an abbreviation or number.
    boundaries = []
    for m in re.finditer(r'[.!?]\s+', text):
        end = m.end()
        start = m.start()

        # Next char must be uppercase (or end of string)
        if end < len(text) and not text[end].isupper():
            continue

        # Check if this is just !? (always a sentence end)
        if text[start] in '!?':
            boundaries.append(end)
            continue

        # It's a period — check the word before it
        pre = text[:start]
        # Single uppercase letter before period (acronym like U.S.)
        if re.search(r'\b[A-Z]$', pre):
            continue
        # Known abbreviation
        if re.search(
            r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Rev|Gen|Gov'
            r'|Sgt|Cpl|Pvt|Lt|Col|Capt|Maj'
            r'|vs|etc|approx|dept|est|inc|corp|ltd|vol|no)$',
            pre, re.IGNORECASE
        ):
            continue
        # Number before period (3.5)
        if re.search(r'\d$', pre):
            continue
        # e.g. / i.e. — period preceded by single lowercase letter
        if re.search(r'\b[a-z]$', pre):
            continue

        boundaries.append(end)

    # Slice text at boundaries
    sentences = []
    prev = 0
    for b in boundaries:
        s = text[prev:b].strip()
        if s:
            sentences.append(s)
        prev = b
    tail = text[prev:].strip()
    if tail:
        sentences.append(tail)

    return sentences if sentences else [text]


def chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """Split text into chunks at natural boundaries.

    Priority: sentence ends (.!?) > clause breaks (,;:—) > word boundary.
    Each chunk stays under max_chars when possible, with a small grace
    window to avoid splitting mid-sentence when a boundary is just past
    the limit.
    """
    import re

    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Pre-split into sentences, then merge into chunks.
    # This ensures we never break mid-sentence unless a single
    # sentence exceeds max_chars.
    sentences = []
    for part in _split_sentences(text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            sentences.append(part)
        else:
            # Single sentence too long — split at clause boundaries
            _split_long_sentence(part, max_chars, sentences)

    # Merge sentences into chunks up to max_chars
    chunks = []
    current = ""
    for sent in sentences:
        if not current:
            current = sent
        elif len(current) + 1 + len(sent) <= max_chars:
            current = current + " " + sent
        else:
            chunks.append(current)
            current = sent
    if current:
        chunks.append(current)

    return chunks


def _split_long_sentence(text: str, max_chars: int, out: list[str]):
    """Split a single long sentence at clause boundaries or word boundaries."""
    import re

    remaining = text.strip()
    while remaining:
        if len(remaining) <= max_chars:
            out.append(remaining)
            break

        # Try clause-level punctuation (comma, semicolon, colon, em-dash)
        best = -1
        for m in re.finditer(r'[,;:\u2014]\s', remaining[:max_chars]):
            best = m.end()

        if best > 0:
            out.append(remaining[:best].strip())
            remaining = remaining[best:].strip()
            continue

        # Fall back to word boundary
        split_at = remaining[:max_chars].rfind(' ')
        if split_at <= 0:
            split_at = max_chars
        out.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()


def adjust_speed(audio: mx.array, speed: float) -> mx.array:
    """Adjust audio playback speed via linear interpolation.

    Args:
        audio: 1D audio waveform
        speed: Speed multiplier (>1.0 = faster, <1.0 = slower)
    """
    if abs(speed - 1.0) < 1e-6:
        return audio
    old_length = audio.shape[0]
    new_length = max(1, int(old_length / speed))
    positions = mx.linspace(0, old_length - 1, new_length)
    left = mx.floor(positions).astype(mx.int32)
    right = mx.minimum(left + 1, old_length - 1)
    frac = positions - left
    return (1.0 - frac) * audio[left] + frac * audio[right]


def generate(model: DualARModel, prompt: mx.array,
             config: GenerationConfig = GenerationConfig(),
             voice_cache: Optional[list] = None,
             text_token_count: Optional[int] = None,
             ) -> Tuple[mx.array, dict]:
    """Generate audio codes from prompt.

    Args:
        voice_cache: Optional pre-filled KV cache from prefill_voice().
            If provided, the cache is restored before prefilling the prompt,
            so the prompt should be just the text suffix (from build_text_suffix).
        text_token_count: Number of text tokens in the input. If provided,
            dynamically caps max_new_tokens to ~12x the text length.
    """
    metrics = {"prefill_ms": 0, "step_ms": 0,
               "total_ms": 0, "n_tokens": 0}

    # Dynamic token budget: ~12 semantic tokens per text token
    max_new_tokens = config.max_new_tokens
    if text_token_count is not None:
        max_new_tokens = min(max_new_tokens, max(32, text_token_count * 12))

    slow_cfg = model.slow_config
    fast_cfg = model.fast_config

    # Pre-compute valid token indices (semantic range + IM_END)
    # This lets us do top-p sampling on ~4097 tokens instead of 155K
    semantic_start = slow_cfg.semantic_start_id
    semantic_end = semantic_start + slow_cfg.codebook_size - 1
    valid_ids = list(range(semantic_start, semantic_end + 1)) + [IM_END]
    valid_indices = mx.array(valid_ids, dtype=mx.int32)

    # Pre-extract output projection weights for valid tokens only.
    # This reduces the per-step matmul from [dim, 155K] to [dim, ~4097] (~38x smaller).
    valid_output_weights = model.slow.tok_embeddings.weight[valid_indices]  # [~4097, dim]

    # Setup KV caches
    # Pre-allocate KV caches sized for prompt + max generation
    est_total = prompt.shape[1] + max_new_tokens + 128
    slow_cache = [
        KVCache(slow_cfg.n_kv_heads, slow_cfg.head_dim, max_len=est_total)
        for _ in range(slow_cfg.n_layers)
    ]
    # Pre-allocated cache for fast AR (max 12 entries: 1 hidden + 11 codebooks)
    fast_cache = [
        RotatingKVCache(fast_cfg.n_kv_heads, fast_cfg.head_dim, max_len=16)
        for _ in range(fast_cfg.n_layers)
    ]

    # Restore voice cache if provided
    if voice_cache is not None:
        for i, (k, v) in enumerate(voice_cache):
            slow_cache[i].set_from(k, v)

    # Prefill
    t0 = time.time()
    prompt_tokens = prompt[0:1, :]  # [1, T]
    prompt_codebooks = prompt[1:, :].transpose(1, 0)[None, :, :]  # [1, T, 10]
    T_prompt = prompt.shape[1]

    # For prefill with existing cache, we need a causal mask that covers
    # both the cached positions and the new prompt
    T_cached = slow_cache[0].offset if voice_cache is not None else 0
    T_total = T_cached + T_prompt
    mask = nn.MultiHeadAttention.create_additive_causal_mask(T_total)
    mask = mask.astype(mx.bfloat16)
    # Only need the rows for the new prompt positions
    mask = mask[T_cached:, :]

    x = model.slow.embed(prompt_tokens, prompt_codebooks)
    # Use projected output even for prefill — only need valid logits from last position
    logits, hidden = model.slow(x, mask=mask, cache=slow_cache,
                                output_weights=valid_output_weights)
    mx.eval(logits, hidden)
    metrics["prefill_ms"] = (time.time() - t0) * 1000

    last_valid_logits = logits[:, -1, :]  # Already projected to [~4097]
    last_hidden = hidden[:, -1:, :]

    # Repetition penalty: pre-allocated ring buffer of mapped token indices.
    # Indices are in valid_ids space (0..codebook_size = semantic, codebook_size = IM_END).
    rep_win = config.repetition_window
    rep_buf = mx.zeros((rep_win,), dtype=mx.int32)
    rep_count = 0  # how many valid entries in buffer
    rep_pos = 0    # next write position (circular)
    use_rep_penalty = config.repetition_penalty != 1.0

    # Pre-allocate output buffer to avoid per-step array creation + final stack
    out_codes = mx.zeros((11, max_new_tokens), dtype=mx.int32)
    n_generated = 0

    # Generation loop
    t_gen = time.time()

    for step in range(max_new_tokens):
        t_step = time.time()

        # === Slow AR: sample semantic token (stays as mx.array) ===
        valid_logits = last_valid_logits[0]  # [~4097]

        # Repetition penalty (fully vectorized, no Python loops)
        if rep_count > 0 and use_rep_penalty:
            n = min(rep_count, rep_win)
            prev_arr = rep_buf[:n]
            scores = mx.take(valid_logits, prev_arr)
            penalty = config.repetition_penalty
            penalized = mx.where(scores < 0, scores * penalty, scores / penalty)
            valid_logits = valid_logits.at[prev_arr].add(penalized - scores)

        token_idx = _sample_token(
            valid_logits[None, :],
            temperature=config.temperature,
            top_p=config.top_p,
        )

        # === Speculative fast AR + slow AR feedback ===
        # Build the full computation graph before eval — token_idx stays as
        # an mx.array so fast AR + slow AR feedback fuse into one eval call.
        # Wastes one fast AR on the final EOS step but eliminates a GPU
        # pipeline stall on every non-EOS step.
        code_0 = mx.clip(token_idx, 0, fast_cfg.codebook_size - 1).squeeze()
        next_vocab_id = valid_indices[token_idx].squeeze()  # Map index -> vocab ID

        for c in fast_cache:
            c.reset()

        codebook_codes = _fast_ar_generate(
            model, last_hidden, code_0, fast_cache, config.temperature
        )

        # Feed back to slow AR
        next_token = next_vocab_id.reshape(1, 1)
        next_codebooks = codebook_codes.reshape(1, 1, 10)

        x = model.slow.embed(next_token, next_codebooks)
        logits, hidden = model.slow(x, cache=slow_cache,
                                    output_weights=valid_output_weights)

        # === Single eval for entire step ===
        mx.eval(token_idx, codebook_codes, logits, hidden)

        idx = token_idx.item()
        token_id = valid_ids[idx]

        if token_id == IM_END:
            break

        # Update repetition penalty ring buffer (idx is already in valid_ids space)
        if use_rep_penalty:
            rep_buf[rep_pos] = idx
            rep_pos = (rep_pos + 1) % rep_win
            rep_count += 1
        metrics["n_tokens"] += 1

        t_now = time.time()
        metrics["step_ms"] += (t_now - t_step) * 1000

        last_valid_logits = logits[:, -1, :]
        last_hidden = hidden[:, -1:, :]

        # Write to pre-allocated buffer (no per-step array creation)
        out_codes[0, n_generated] = token_id
        out_codes[1:, n_generated] = codebook_codes
        n_generated += 1

    metrics["total_ms"] = (time.time() - t_gen) * 1000

    if n_generated == 0:
        return mx.zeros((10, 0), dtype=mx.int32), metrics

    codes = out_codes[1:, :n_generated]

    return codes, metrics
