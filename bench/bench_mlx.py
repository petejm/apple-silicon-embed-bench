#!/usr/bin/env python3
"""MLX-embeddings bench — isolated from coremltools. Apple silicon Metal via MLX."""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus", "corpus_buckets.json")
OUT = os.path.join(ROOT, "results", "mlx_embeddings.json")


def main():
    from mlx_embeddings.utils import load, generate
    import mlx.core as mx
    materialize = mx.eval  # MLX tensor materialization; see README "MLX is lazy"
    # load() takes no revision arg in mlx-embeddings 0.1.0; pin by downloading
    # the exact revision via snapshot_download and passing the returned path
    # directly to load(). Capturing the path is load-bearing: if the HF cache
    # already has a different revision (from another tool's prior install),
    # `load("BAAI/bge-small-en-v1.5")` would silently use whatever is on disk
    # rather than the revision we just pinned.
    from huggingface_hub import snapshot_download
    snapshot_path = snapshot_download(
        "BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    )
    model, tokenizer = load(snapshot_path)
    # Surface the runtime dtype so the result block can prove FP16 (and not
    # silently fall back to FP32). MLX exposes parameters() as a dict-tree;
    # walk to the first leaf tensor and print its dtype.
    def _first_param_dtype(params):
        if hasattr(params, "dtype"):
            return params.dtype
        if isinstance(params, dict):
            for v in params.values():
                d = _first_param_dtype(v)
                if d is not None:
                    return d
        if isinstance(params, (list, tuple)):
            for v in params:
                d = _first_param_dtype(v)
                if d is not None:
                    return d
        return None
    try:
        first_dtype = _first_param_dtype(model.parameters())
        first_dtype_str = str(first_dtype) if first_dtype is not None else "unknown"
    except Exception as e:
        first_dtype_str = f"introspect-failed: {type(e).__name__}: {e}"
    print(f"model loaded; first-param dtype: {first_dtype_str}", flush=True)
    # Soft assertion: warn (don't crash) if not float16. Comparing against
    # the bench's documented contract.
    if "float16" not in first_dtype_str:
        print(f"WARN: MLX first-param dtype is {first_dtype_str}, not float16. "
              "Headline numbers assume FP16; cross-backend parity check (verify_parity.py) "
              "is the canonical apples-to-apples cosine.", flush=True)
    results_dtype = first_dtype_str

    with open(CORPUS) as f:
        buckets = json.load(f)
    results = {"backend": "MLX-embeddings (auto compute, mostly Metal GPU)",
               "model": "bge-small-en-v1.5",
               "first_param_dtype": results_dtype,
               "buckets": {}}

    t0 = time.perf_counter()
    out = generate(model, tokenizer, [buckets["short"][0]])
    materialize(out.text_embeds)
    print("first embed shape:", out.text_embeds.shape, "first 3 vals:", out.text_embeds[0,:3].tolist())
    cold = time.perf_counter() - t0
    results["cold_start_s"] = cold
    print(f"cold {cold:.3f}s", flush=True)

    for bucket, sentences in buckets.items():
        runs_b1 = []
        for r in range(5):
            t = time.perf_counter()
            for s in sentences:
                out = generate(model, tokenizer, [s])
                materialize(out.text_embeds)
            elapsed = time.perf_counter() - t
            runs_b1.append(len(sentences) / elapsed)

        runs_bN = []
        for r in range(5):
            t = time.perf_counter()
            out = generate(model, tokenizer, sentences)
            materialize(out.text_embeds)
            elapsed = time.perf_counter() - t
            runs_bN.append(len(sentences) / elapsed)

        results["buckets"][bucket] = {
            "batch1_sent_per_s_runs": runs_b1,
            "batch1_sent_per_s": float(np.mean(runs_b1[1:])),
            "batchN_sent_per_s_runs": runs_bN,
            "batchN_sent_per_s": float(np.mean(runs_bN[1:])),
        }
        print(f"[{bucket}] b=1: {results['buckets'][bucket]['batch1_sent_per_s']:.1f} sent/s, b=100: {results['buckets'][bucket]['batchN_sent_per_s']:.1f} sent/s", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
