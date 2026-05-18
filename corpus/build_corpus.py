#!/usr/bin/env python3
"""Build a deterministic, public-domain corpus for the embedding bench.

Source: Project Gutenberg "Pride and Prejudice" body excerpt (first ~80KB).
Public domain in the US and most jurisdictions; Project Gutenberg header
and footer stripped per their license. See corpus/NOTICE.md for the
provenance / license details.

No network dependency: the text is bundled in `sources.txt`.

Output: corpus_buckets.json with three buckets:
  - short (~32 tokens avg)
  - medium (~124 tokens avg)
  - long (~462 tokens avg)
100 items per bucket.

Reproducibility: deterministic given the bundled source. Re-running produces
byte-identical output. This matters for cross-machine comparisons.
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = HERE / "sources.txt"
OUT = HERE / "corpus_buckets.json"

# Rough char-per-token ratio for English with a BPE tokenizer.
# We bucket by char count proxies, not exact tokens, to avoid pulling
# a tokenizer dependency into the corpus build step. The bench scripts
# do real tokenization at run time.
CHARS_PER_TOKEN = 4

BUCKET_TARGETS = {
    "short": (16 * CHARS_PER_TOKEN, 48 * CHARS_PER_TOKEN),    # ~32 tok
    "medium": (96 * CHARS_PER_TOKEN, 160 * CHARS_PER_TOKEN),  # ~124 tok
    "long": (380 * CHARS_PER_TOKEN, 540 * CHARS_PER_TOKEN),   # ~462 tok
}
BUCKET_SIZE = 100


def split_sentences(text):
    """Naive sentence splitter — good enough for English public-domain prose."""
    text = re.sub(r"\s+", " ", text).strip()
    # Split on sentence terminators but keep them attached
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text_to_length(text, target_lo, target_hi):
    """Greedy chunker: join sentences until we hit the target length window."""
    sentences = split_sentences(text)
    chunks = []
    buf = []
    buf_len = 0
    for s in sentences:
        s_len = len(s)
        if buf_len + s_len + 1 > target_hi and buf:
            chunks.append(" ".join(buf))
            buf = []
            buf_len = 0
        buf.append(s)
        buf_len += s_len + 1
        if buf_len >= target_lo:
            chunks.append(" ".join(buf))
            buf = []
            buf_len = 0
    if buf and buf_len >= target_lo:
        chunks.append(" ".join(buf))
    return chunks


def cycle_pad(items, n):
    """Repeat items deterministically to reach length n."""
    out = []
    while len(out) < n:
        out.extend(items)
    return out[:n]


def main():
    if not SOURCES.exists():
        print(f"ERROR: {SOURCES} not found — corpus build cannot proceed", file=sys.stderr)
        sys.exit(1)

    raw = SOURCES.read_text(encoding="utf-8")
    # Skip provenance header (lines starting with #) — those describe license
    # status of the bundled text, not corpus content.
    text = "\n".join(ln for ln in raw.splitlines() if not ln.lstrip().startswith("#"))
    buckets = {}
    for name, (lo, hi) in BUCKET_TARGETS.items():
        chunks = chunk_text_to_length(text, lo, hi)
        if len(chunks) < BUCKET_SIZE:
            chunks = cycle_pad(chunks, BUCKET_SIZE)
        else:
            chunks = chunks[:BUCKET_SIZE]
        buckets[name] = chunks
        avg_len = sum(len(c) for c in chunks) / len(chunks)
        print(f"{name}: {len(chunks)} chunks, avg {avg_len:.0f} chars (~{avg_len/CHARS_PER_TOKEN:.0f} tok)")

    OUT.write_text(json.dumps(buckets, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")

    # Also write per-bucket plain-text files for llama-embedding's -f flag.
    # One sentence per line. llama-embedding processes each line as one prompt.
    for name, items in buckets.items():
        f = HERE / f"corpus_llama_{name}.txt"
        f.write_text("\n".join(s.replace("\n", " ").strip() for s in items) + "\n")
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
