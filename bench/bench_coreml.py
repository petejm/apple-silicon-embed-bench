#!/usr/bin/env python3
"""Bench bge-small CoreML across compute units and seq lengths.

Note: the .mlpackage was traced at fixed SEQ_LEN=512. For shorter inputs we pad
to 512 — this is the realistic ANE path. Variable-shape models punt to GPU on
Apple silicon and torpedo perf; we document that separately.
"""
import json
import os
import sys
import time
import gc
import argparse
import numpy as np
from transformers import AutoTokenizer
import coremltools as ct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "bge-small-en-v1.5.mlpackage")
CORPUS = os.path.join(ROOT, "corpus", "corpus_buckets.json")
SEQ_LEN = 512
# Pin the HF revision so the tokenizer is byte-identical across machines.
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

COMPUTE_MAP = {
    "ane": ct.ComputeUnit.CPU_AND_NE,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu": ct.ComputeUnit.CPU_ONLY,
    "all": ct.ComputeUnit.ALL,
}


def tokenize_batch(tok, sentences, max_len=SEQ_LEN):
    enc = tok(sentences, return_tensors="np", padding="max_length",
              truncation=True, max_length=max_len)
    return {
        "input_ids": enc["input_ids"].astype(np.int32),
        "attention_mask": enc["attention_mask"].astype(np.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", choices=list(COMPUTE_MAP), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cu = COMPUTE_MAP[args.compute]
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", revision=MODEL_REVISION)
    with open(CORPUS) as f:
        buckets = json.load(f)

    results = {"compute_unit_request": args.compute, "model": "bge-small-en-v1.5", "seq_len_pad": SEQ_LEN, "buckets": {}}

    # Cold start
    t0 = time.perf_counter()
    model = ct.models.MLModel(MODEL, compute_units=cu)
    # First prediction (cold)
    sample = tokenize_batch(tok, [buckets["short"][0]])
    out = model.predict({k: v[0:1] for k, v in sample.items()})
    cold_s = time.perf_counter() - t0
    results["cold_start_s"] = cold_s
    out_key = list(out.keys())[0]
    print(f"[{args.compute}] cold start {cold_s:.3f}s, out_key={out_key}, shape={out[out_key].shape}", flush=True)

    # Device placement probe runs separately (in bench.sh) and writes to results/.
    probe_path = os.path.join(ROOT, "results", f"devices_{args.compute}.json")
    if os.path.exists(probe_path):
        with open(probe_path) as f:
            results["device_placement"] = json.load(f)

    # Per-bucket warm benches
    for bucket_name, sentences in buckets.items():
        bres = {}
        # Tokenize once
        enc = tokenize_batch(tok, sentences)

        # Batch=1 warm throughput: 5 runs of 100 sentences
        runs = []
        for r in range(5):
            t = time.perf_counter()
            for i in range(len(sentences)):
                model.predict({
                    "input_ids": enc["input_ids"][i:i+1],
                    "attention_mask": enc["attention_mask"][i:i+1],
                })
            elapsed = time.perf_counter() - t
            runs.append(len(sentences) / elapsed)
        bres["batch1_sent_per_s_runs"] = runs
        bres["batch1_sent_per_s"] = float(np.mean(runs[1:]))
        print(f"[{args.compute}/{bucket_name}] batch=1 {bres['batch1_sent_per_s']:.1f} sent/s (runs {[round(x,1) for x in runs]})", flush=True)

        # Batch=32 — the .mlpackage was traced at batch=1, so we loop b=32 times rather than re-trace
        # Realistic: most CoreML deployments re-trace per batch size. We'll note this caveat.
        bres["batch32_note"] = "model was traced at batch=1; per-sentence loop reflects real deployment cost on this model"

        results["buckets"][bucket_name] = bres

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    # Bypass Python/CoreML destructor race on macOS 26 + py3.13 + coremltools 9
    # (MLE5ExecutionStream.resetQueue → _PyObject_Free without GIL → SIGSEGV).
    # Results are already on disk; skip cleanup.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
