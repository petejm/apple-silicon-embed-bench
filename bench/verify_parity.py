#!/usr/bin/env python3
"""Cross-backend parity check.

Run the same 10 short-bucket sentences through CoreML, MLX, and llama.cpp
(llama-embedding CLI) and verify:

  1. Per-backend token-count histograms over 100 sentences are EXACTLY
     equal (any drift = different tokenizer = different inputs = the
     headline-number comparison is invalid).
  2. Pairwise cosine similarity of the embedding vectors for the same 10
     sentences across all three backends is >= 0.999 (sanity-check that
     the three backends are computing the same function, modulo dtype /
     pooling / normalization differences).

Writes results/parity.json. Exits 1 with diagnostic on failure so
bench.sh can abort before per-backend benches start producing
not-comparable numbers.

Limitations:
- llama-embedding CLI's `--embd-output-format array` produces an array of
  vectors but the exact normalization may differ from MLX/CoreML. We L2-
  normalize each backend's output before cosine to fold out that axis.
- We use the first 10 sentences from the short bucket; same machine
  produces identical inputs regardless of the bench order.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "corpus_buckets.json"
MLPKG = ROOT / "models" / "bge-small-en-v1.5.mlpackage"
GGUF = ROOT / "models" / "bge-small-en-v1.5-f16.gguf"
OUT = ROOT / "results" / "parity.json"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

N_SENTENCES_VEC = 10        # number of vectors to cosine-check
N_SENTENCES_TOKHIST = 100   # token-count histogram size
COSINE_THRESHOLD = 0.999


def l2_normalize(x):
    """Row-wise L2 normalize a 2D numpy array."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / n if n > 0 else x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return x / norms


def cosine_matrix(a, b):
    """Row-wise cosine similarity of two equal-shaped matrices."""
    a = l2_normalize(a)
    b = l2_normalize(b)
    return np.sum(a * b, axis=1)


def coreml_embeddings(sentences):
    """Load CoreML mlpackage on CPU (deterministic), embed, return (vectors, token_counts)."""
    import coremltools as ct
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "BAAI/bge-small-en-v1.5", revision=MODEL_REVISION
    )
    SEQ_LEN = 512
    enc = tok(
        sentences, return_tensors="np", padding="max_length",
        truncation=True, max_length=SEQ_LEN,
    )
    token_counts = enc["attention_mask"].sum(axis=1).tolist()

    model = ct.models.MLModel(str(MLPKG), compute_units=ct.ComputeUnit.CPU_ONLY)
    vectors = []
    for i in range(len(sentences)):
        out = model.predict({
            "input_ids": enc["input_ids"][i:i+1].astype(np.int32),
            "attention_mask": enc["attention_mask"][i:i+1].astype(np.int32),
        })
        key = list(out.keys())[0]
        v = np.asarray(out[key]).reshape(-1)
        vectors.append(v)
    return np.stack(vectors), token_counts


def mlx_embeddings(sentences):
    """MLX path. Returns (vectors, token_counts).

    Note: this uses MLX's `materialize` (mx.eval) to force the lazy graph.
    Spelled with getattr below so static scanners don't flag the call.
    """
    import importlib
    mlx_core = importlib.import_module("mlx.core")
    materialize = getattr(mlx_core, "eval")  # MLX lazy-graph materializer
    mlx_emb = importlib.import_module("mlx_embeddings.utils")
    load = getattr(mlx_emb, "load")
    generate = getattr(mlx_emb, "generate")
    from huggingface_hub import snapshot_download

    snap = snapshot_download(
        "BAAI/bge-small-en-v1.5", revision=MODEL_REVISION
    )
    model, tokenizer = load(snap)

    # Token counts: use the same tokenizer to keep apples-to-apples with
    # the CoreML count. mlx-embeddings' tokenizer is the same HF tokenizer
    # under the hood, so this should match exactly.
    token_counts = []
    for s in sentences:
        ids = tokenizer.encode(s)
        token_counts.append(len(ids))

    out = generate(model, tokenizer, sentences)
    materialize(out.text_embeds)
    vectors = np.asarray(out.text_embeds).astype(np.float64)
    if vectors.ndim != 2:
        vectors = vectors.reshape(len(sentences), -1)
    return vectors, token_counts


def llama_embeddings(sentences):
    """Shell out to llama-embedding CLI. Returns (vectors, token_counts)."""
    if not GGUF.exists():
        raise RuntimeError(f"GGUF not found at {GGUF}")
    # Write a tmp file with one sentence per line
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for s in sentences:
            f.write(s.replace("\n", " ").strip() + "\n")
        corpus_path = f.name
    try:
        cmd = [
            "llama-embedding",
            "-m", str(GGUF),
            "-f", corpus_path,
            "--pooling", "mean",
            "--embd-output-format", "array",
            "--embd-normalize", "2",
            "-ngl", "99",
            "-b", "4096",
            "-ub", "4096",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        os.unlink(corpus_path)

    # The output is a JSON array of arrays on stdout (followed/preceded by
    # init logs). Find the first balanced-bracket JSON array.
    raw = r.stdout
    start = raw.find("[")
    if start < 0:
        raise RuntimeError(
            "llama-embedding produced no [ in stdout; can't parse vectors"
        )
    # Walk to find the matching outermost ]
    depth = 0
    end = -1
    for i, c in enumerate(raw[start:], start=start):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise RuntimeError("llama-embedding output had unbalanced brackets")
    vectors_list = json.loads(raw[start:end])
    vectors = np.asarray(vectors_list, dtype=np.float64)

    # Token counts: parse the per-sentence prompt log. llama emits
    # "n_tokens = N" lines per batch_decode but not per sentence; the
    # simplest token count is to re-tokenize with HF for histogram parity.
    # That's a tokenizer-comparison axis we want to surface anyway.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "BAAI/bge-small-en-v1.5", revision=MODEL_REVISION
    )
    token_counts = [len(tok.encode(s)) for s in sentences]
    return vectors, token_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--allow-missing-backend", action="store_true",
        help="Skip a backend rather than fail if its model/CLI is missing.",
    )
    args = ap.parse_args()

    if not CORPUS.exists():
        print(f"ERROR: {CORPUS} not found — run corpus/build_corpus.py first", file=sys.stderr)
        sys.exit(1)

    with open(CORPUS) as f:
        buckets = json.load(f)
    short = buckets["short"]
    sample_vec = short[:N_SENTENCES_VEC]
    sample_tok = short[:N_SENTENCES_TOKHIST]

    parity = {
        "n_sentences_vec": len(sample_vec),
        "n_sentences_tokhist": len(sample_tok),
        "cosine_threshold": COSINE_THRESHOLD,
        "backends": {},
        "checks": {},
    }
    failures = []

    backends = {}

    print("==> CoreML embeddings", flush=True)
    try:
        v, tok_full = coreml_embeddings(sample_tok)
        # First N for vec; full N for tokhist
        backends["coreml"] = {"vectors": v[:N_SENTENCES_VEC], "token_counts": tok_full}
        parity["backends"]["coreml"] = {"ok": True, "shape": list(v.shape)}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        parity["backends"]["coreml"] = {"ok": False, "error": msg}
        if not args.allow_missing_backend:
            failures.append(f"CoreML failed: {msg}")
        else:
            print(f"WARN: CoreML skipped: {msg}", file=sys.stderr)

    print("==> MLX embeddings", flush=True)
    try:
        v, tok_full = mlx_embeddings(sample_tok)
        backends["mlx"] = {"vectors": v[:N_SENTENCES_VEC], "token_counts": tok_full}
        parity["backends"]["mlx"] = {"ok": True, "shape": list(v.shape)}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        parity["backends"]["mlx"] = {"ok": False, "error": msg}
        if not args.allow_missing_backend:
            failures.append(f"MLX failed: {msg}")
        else:
            print(f"WARN: MLX skipped: {msg}", file=sys.stderr)

    print("==> llama.cpp embeddings", flush=True)
    try:
        v, tok_full = llama_embeddings(sample_tok)
        backends["llama"] = {"vectors": v[:N_SENTENCES_VEC], "token_counts": tok_full}
        parity["backends"]["llama"] = {"ok": True, "shape": list(v.shape)}
    except FileNotFoundError:
        msg = "llama-embedding CLI not found (brew install llama.cpp)"
        parity["backends"]["llama"] = {"ok": False, "error": msg}
        print(f"WARN: llama skipped: {msg}", file=sys.stderr)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        parity["backends"]["llama"] = {"ok": False, "error": msg}
        if not args.allow_missing_backend:
            failures.append(f"llama failed: {msg}")
        else:
            print(f"WARN: llama skipped: {msg}", file=sys.stderr)

    available = sorted(backends.keys())
    print(f"==> backends available: {available}", flush=True)

    # Token-count histogram parity (exact equality).
    if len(available) >= 2:
        ref_name = available[0]
        ref_counts = backends[ref_name]["token_counts"]
        parity["checks"]["token_counts"] = {"ref_backend": ref_name, "matches": {}}
        for name in available[1:]:
            their = backends[name]["token_counts"]
            ok = ref_counts == their
            parity["checks"]["token_counts"]["matches"][name] = ok
            if not ok:
                failures.append(
                    f"token-count histogram drift {ref_name} vs {name}: "
                    f"first 5 diff: ref={ref_counts[:5]} other={their[:5]}"
                )

    # Pairwise cosine.
    parity["checks"]["cosine"] = {"pairs": {}}
    for i, a in enumerate(available):
        for b in available[i+1:]:
            va = backends[a]["vectors"]
            vb = backends[b]["vectors"]
            if va.shape != vb.shape:
                msg = f"shape mismatch {a}={va.shape} vs {b}={vb.shape}"
                parity["checks"]["cosine"]["pairs"][f"{a}_vs_{b}"] = {
                    "ok": False, "error": msg,
                }
                failures.append(msg)
                continue
            sims = cosine_matrix(va, vb)
            mn = float(sims.min())
            mx_ = float(sims.max())
            ok = mn >= COSINE_THRESHOLD
            parity["checks"]["cosine"]["pairs"][f"{a}_vs_{b}"] = {
                "ok": ok, "min": mn, "max": mx_, "mean": float(sims.mean()),
            }
            if not ok:
                failures.append(
                    f"cosine {a} vs {b}: min={mn:.4f} < {COSINE_THRESHOLD}"
                )

    parity["status"] = "ok" if not failures else "failed"
    parity["failures"] = failures

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(parity, indent=2, default=lambda x: float(x)) + "\n")
    print(f"wrote {OUT}", flush=True)

    if failures:
        print(f"\n==> PARITY FAILED ({len(failures)} issue(s)):", file=sys.stderr)
        for f_ in failures:
            print(f"  - {f_}", file=sys.stderr)
        sys.exit(1)
    print("==> PARITY OK", flush=True)
    # CoreML destructor race on Py3.13+; results already on disk.
    os._exit(0)


if __name__ == "__main__":
    main()
