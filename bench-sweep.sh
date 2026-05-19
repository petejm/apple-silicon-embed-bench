#!/usr/bin/env bash
# Outer-loop wrapper for reproducing the canonical 10-run-mean numbers.
#
# `./bench.sh` is a single invocation: 5 inner runs × 100 sentences per
# bucket, drop run 1, mean of runs 2-5. That gives a noisy point estimate,
# NOT the headline number in the README.
#
# The README headline cells are the mean of 10 outer-run means — i.e.,
# 10 independent invocations of `./bench.sh`. This script wraps that loop.
#
# Usage:
#   N_SWEEPS=10 ./bench-sweep.sh        # canonical: 10 outer runs
#   N_SWEEPS=3  ./bench-sweep.sh        # quick variance probe
#
# Output:
#   results/run-01/  results/run-02/  ...  results/run-NN/
#   results/sweep_summary.json   — aggregated mean / range / σ / CV per cell
#   results/sweep_summary.md     — human-readable table
#
# This does NOT change `./bench.sh` behavior; it just runs it in a loop.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

N_SWEEPS="${N_SWEEPS:-10}"
if ! [[ "$N_SWEEPS" =~ ^[0-9]+$ ]] || [ "$N_SWEEPS" -lt 1 ]; then
  echo "ERROR: N_SWEEPS must be a positive integer (got: $N_SWEEPS)" >&2
  exit 1
fi

echo "==> bench-sweep: running ./bench.sh ${N_SWEEPS} times"
echo "==> each invocation produces a point estimate; the aggregate is the canonical number"
echo

mkdir -p results

for i in $(seq 1 "$N_SWEEPS"); do
  nn=$(printf "%02d" "$i")
  echo
  echo "================================================================"
  echo "==> sweep run ${nn} / ${N_SWEEPS}"
  echo "================================================================"
  # Suppress check.sh on inner runs after the first — they're redundant.
  if [ "$i" -gt 1 ]; then
    SKIP_CHECK=1 ./bench.sh
  else
    ./bench.sh
  fi
  # Snapshot this run's results into results/run-NN/
  dest="results/run-${nn}"
  rm -rf "$dest"
  mkdir -p "$dest"
  # Copy all JSONs + RESULT.md, leave the top-level results/ alone for the
  # next run (bench.sh overwrites in place).
  for f in results/*.json results/RESULT.md; do
    [ -f "$f" ] && cp "$f" "$dest/"
  done
done

echo
echo "================================================================"
echo "==> aggregating ${N_SWEEPS} runs"
echo "================================================================"

python3 - "$N_SWEEPS" <<'PY'
import json
import sys
from pathlib import Path

N = int(sys.argv[1])
R = Path("results")

# Build the cell matrix: backend × bucket × sweep-run → per-invocation mean
def load(run_dir, name):
    p = run_dir / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def inner_mean(runs):
    """Drop run 1, mean of last (len-1) runs. Same protocol as bench.sh's _fmt_stats."""
    if not runs or len(runs) < 2:
        return runs[0] if runs else None
    vals = runs[1:]
    return sum(vals) / len(vals)

def coreml_cell(d, bucket, mode):
    """mode: 'b1' for batch1, 'bN' for batchN. CoreML has only b1."""
    if not d or mode != 'b1':
        return None
    try:
        runs = d['buckets'][bucket].get('batch1_sent_per_s_runs')
        if runs:
            return inner_mean(runs)
        return d['buckets'][bucket].get('batch1_sent_per_s')
    except (KeyError, TypeError):
        return None

def mlx_cell(d, bucket, mode):
    if not d:
        return None
    try:
        key = 'batch1_sent_per_s_runs' if mode == 'b1' else 'batchN_sent_per_s_runs'
        runs = d['buckets'][bucket].get(key)
        if runs:
            return inner_mean(runs)
        scalar = 'batch1_sent_per_s' if mode == 'b1' else 'batchN_sent_per_s'
        return d['buckets'][bucket].get(scalar)
    except (KeyError, TypeError):
        return None

def llama_cell(d, bucket, mode):
    if not d or mode != 'bN':
        return None
    try:
        runs = [r['sent_per_s_total'] for r in d['buckets'][bucket]['runs']
                if r.get('status', 'ok') == 'ok' and 'sent_per_s_total' in r]
        return inner_mean(runs) if len(runs) >= 2 else None
    except (KeyError, TypeError):
        return None

BACKENDS = [
    ("CoreML ANE", "coreml_ane.json", coreml_cell),
    ("CoreML GPU", "coreml_gpu.json", coreml_cell),
    ("CoreML CPU", "coreml_cpu.json", coreml_cell),
    ("MLX-embeddings", "mlx_embeddings.json", mlx_cell),
    ("llama.cpp Metal", "llama_metal.json", llama_cell),
]
BUCKETS = ["short", "medium", "long"]
MODES = ["b1", "bN"]

# collect: results[label][bucket][mode] = [point, point, ...]
collected = {b[0]: {bk: {m: [] for m in MODES} for bk in BUCKETS} for b in BACKENDS}
for i in range(1, N + 1):
    run_dir = R / f"run-{i:02d}"
    for label, fname, fn in BACKENDS:
        d = load(run_dir, fname)
        for bk in BUCKETS:
            for m in MODES:
                v = fn(d, bk, m)
                if v is not None and v > 0:
                    collected[label][bk][m].append(v)

def agg(vals):
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    lo, hi = min(vals), max(vals)
    if n > 1:
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        sigma = var ** 0.5
        cv = 100.0 * sigma / mean if mean > 0 else 0
        return {"n": n, "mean": mean, "min": lo, "max": hi, "sigma": sigma, "cv_pct": cv}
    return {"n": n, "mean": mean, "min": lo, "max": hi, "sigma": 0.0, "cv_pct": 0.0}

# JSON output
summary = {}
for label, _, _ in BACKENDS:
    summary[label] = {}
    for bk in BUCKETS:
        summary[label][bk] = {}
        for m in MODES:
            a = agg(collected[label][bk][m])
            if a:
                summary[label][bk][m] = a

(R / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
print(f"wrote {R / 'sweep_summary.json'}")

# Markdown table
def fmt(a):
    if not a:
        return "—"
    return f"{a['mean']:.0f} [{a['min']:.0f}…{a['max']:.0f}] σ={a['sigma']:.0f} ({a['cv_pct']:.1f}%)"

lines = [
    f"# Sweep summary — {N} runs",
    "",
    "Sentences/sec, mean [min…max] σ=N (CV%). Each cell is the mean of N outer-run inner-means.",
    "",
    "| backend | short b=1 | medium b=1 | long b=1 | short batched | medium batched | long batched |",
    "|---|---|---|---|---|---|---|",
]
for label, _, _ in BACKENDS:
    row = [
        fmt(summary[label]["short"].get("b1")),
        fmt(summary[label]["medium"].get("b1")),
        fmt(summary[label]["long"].get("b1")),
        fmt(summary[label]["short"].get("bN")),
        fmt(summary[label]["medium"].get("bN")),
        fmt(summary[label]["long"].get("bN")),
    ]
    lines.append(f"| {label} | " + " | ".join(row) + " |")

(R / "sweep_summary.md").write_text("\n".join(lines) + "\n")
print(f"wrote {R / 'sweep_summary.md'}")
print()
print("\n".join(lines))
PY

echo
echo "==> sweep complete. Per-run data in results/run-NN/, aggregate in results/sweep_summary.{json,md}"
