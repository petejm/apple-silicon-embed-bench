#!/usr/bin/env bash
# One-command bench runner: builds corpus, converts model to CoreML if needed,
# downloads the GGUF if missing, runs all four backends, prints a result block
# you can paste into a GitHub issue.
#
# Usage: ./bench.sh
# Prereqs: macOS, Python 3.11 or 3.12 (3.13 has a coremltools+CoreML crash bug),
#          Homebrew, llama.cpp installed via `brew install llama.cpp`.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> apple-silicon-embed-bench"
echo "==> repo:  $ROOT"
echo "==> macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
echo "==> chip:  $(sysctl -n machdep.cpu.brand_string)"
echo "==> mem:   $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB"
echo

# 1. Python env
if [ ! -d venv ]; then
  echo "==> creating venv"
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 2. Build corpus (deterministic; ~1 sec)
if [ ! -f corpus/corpus_buckets.json ]; then
  echo "==> building corpus"
  python3 corpus/build_corpus.py
fi

# 3. Convert bge-small to CoreML mlpackage (one-time, ~30 sec)
if [ ! -d models/bge-small-en-v1.5.mlpackage ]; then
  echo "==> converting bge-small to CoreML"
  mkdir -p models
  python3 bench/convert_bge_coreml.py
fi

# 4. Download GGUF for llama.cpp (one-time, ~67 MB)
GGUF=models/bge-small-en-v1.5-f16.gguf
if [ ! -f "$GGUF" ]; then
  echo "==> downloading bge-small GGUF from HuggingFace"
  curl -L --fail -o "$GGUF" \
    "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f16.gguf"
fi

mkdir -p results

# 5. Run benches
echo
echo "==> CoreML ANE bench"
python3 bench/bench_coreml.py --compute ane --out results/coreml_ane.json || true
echo "==> CoreML GPU bench"
python3 bench/bench_coreml.py --compute gpu --out results/coreml_gpu.json || true
echo "==> CoreML CPU bench"
python3 bench/bench_coreml.py --compute cpu --out results/coreml_cpu.json || true
echo "==> llama.cpp Metal bench"
if command -v llama-embedding >/dev/null 2>&1; then
  bench/bench_llama_v2.sh
else
  echo "WARN: llama-embedding not found. Install with: brew install llama.cpp"
fi
echo "==> MLX-embeddings bench"
python3 bench/bench_mlx.py || true

# 6. Aggregate + print pasteable result block
echo
echo "================================================================"
echo "RESULT BLOCK (copy everything between BEGIN/END into your issue)"
echo "================================================================"
echo "===BEGIN RESULT==="
python3 - <<'PY'
import json, os, platform, subprocess, sys
from pathlib import Path

R = Path("results")
def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None

results = {
    "coreml_ane": load("coreml_ane.json"),
    "coreml_gpu": load("coreml_gpu.json"),
    "coreml_cpu": load("coreml_cpu.json"),
    "llama_metal": load("llama_metal.json"),
    "mlx": load("mlx_embeddings.json"),
}

# Hardware
chip = subprocess.run(["sysctl","-n","machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
mem_gb = int(int(subprocess.run(["sysctl","-n","hw.memsize"], capture_output=True, text=True).stdout) / 1024**3)
macos = subprocess.run(["sw_vers","-productVersion"], capture_output=True, text=True).stdout.strip()
build = subprocess.run(["sw_vers","-buildVersion"], capture_output=True, text=True).stdout.strip()
try:
    llama_v = subprocess.run(["llama-embedding","--version"], capture_output=True, text=True).stderr.split("\n")[0]
except Exception:
    llama_v = "n/a"

print(f"## bench result")
print(f"chip: {chip}")
print(f"memory: {mem_gb} GB")
print(f"macOS: {macos} ({build})")
print(f"python: {sys.version.split()[0]}")
print(f"llama.cpp: {llama_v}")
print()
print("| backend | short b=1 | short batched | medium b=1 | medium batched | long b=1 | long batched | cold (s) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")

def b1(d, b):
    try: return f"{d['buckets'][b]['batch1_sent_per_s']:.1f}"
    except: return "—"
def bN(d, b):
    try: return f"{d['buckets'][b]['batchN_sent_per_s']:.1f}"
    except: return "—"
def cold(d):
    try: return f"{d['cold_start_s']:.2f}"
    except: return "—"
def llama_total(d, b):
    try:
        runs = d['buckets'][b]['runs']
        vals = [r['sent_per_s_total'] for r in runs[1:]]
        return f"{sum(vals)/len(vals):.1f}"
    except: return "—"

for label, key in [("CoreML ANE","coreml_ane"),("CoreML GPU","coreml_gpu"),("CoreML CPU","coreml_cpu")]:
    d = results[key]
    print(f"| {label} | {b1(d,'short')} | — | {b1(d,'medium')} | — | {b1(d,'long')} | — | {cold(d)} |")

d = results['mlx']
print(f"| MLX-embeddings | {b1(d,'short')} | {bN(d,'short')} | {b1(d,'medium')} | {bN(d,'medium')} | {b1(d,'long')} | {bN(d,'long')} | {cold(d)} |")

d = results['llama_metal']
print(f"| llama.cpp Metal | — | {llama_total(d,'short')} | — | {llama_total(d,'medium')} | — | {llama_total(d,'long')} | — |")
print()
print("Sentences/sec, higher is better. CoreML mlpackage traced at batch=1 seq=512 fixed.")
print("MLX `b=1` = sentence-by-sentence; `batched` = all 100 in one call. llama.cpp internally batches (~66 seq/forward).")
PY
echo "===END RESULT==="
echo
echo "Submit your numbers at https://github.com/petejm/apple-silicon-embed-bench/issues/new?template=bench-result.md"
