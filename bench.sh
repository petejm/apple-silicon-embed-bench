#!/usr/bin/env bash
# One-command bench runner: builds corpus, fetches the converted CoreML
# mlpackage from the GitHub release (or converts locally if you set
# REBUILD_MLPACKAGE=1), downloads the GGUF, runs all backends, prints a
# pasteable result block.
#
# Prereqs: macOS, Python 3.11 or 3.12 (3.13 has a coremltools crash; bench
# works around via os._exit(0) but 3.12 is recommended). Optional:
# `brew install llama.cpp` for the llama.cpp row.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> apple-silicon-embed-bench"
echo "==> repo:  $ROOT"
echo "==> macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
echo "==> chip:  $(sysctl -n machdep.cpu.brand_string)"
# Use bc for proper rounding so 24GB doesn't print as 23GB
mem_gb=$(python3 -c "print(round($(sysctl -n hw.memsize) / 1024**3))")
echo "==> mem:   ${mem_gb} GB"
echo

# Track per-backend status so the result block can flag failures explicitly.
# macOS ships bash 3.2 which lacks associative arrays; use a temp status file.
STATUS_FILE="$(mktemp)"
trap 'rm -f "$STATUS_FILE"' EXIT
set_status() { printf "%s\t%s\n" "$1" "$2" >> "$STATUS_FILE"; }
get_statuses_json() {
  python3 -c '
import json, sys
d = {}
for line in open(sys.argv[1]):
    k, _, v = line.rstrip("\n").partition("\t")
    if k: d[k] = v
print(json.dumps(d))
' "$STATUS_FILE"
}

# Computed at runtime so forks don't lie to their users
REMOTE_URL=$(git config --get remote.origin.url 2>/dev/null || echo "")
if [[ "$REMOTE_URL" == *"github.com"* ]]; then
  case "$REMOTE_URL" in
    git@github.com:*) GH_REPO="${REMOTE_URL#git@github.com:}"; GH_REPO="${GH_REPO%.git}" ;;
    https://github.com/*) GH_REPO="${REMOTE_URL#https://github.com/}"; GH_REPO="${GH_REPO%.git}" ;;
    *) GH_REPO="petejm/apple-silicon-embed-bench" ;;
  esac
else
  GH_REPO="petejm/apple-silicon-embed-bench"
fi

# Pinned artifact SHAs — every machine should compute against byte-identical inputs.
GGUF_SHA256="f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999"
MLPACKAGE_TARBALL_SHA256="3215e826fbc4adf6710faefa650d3149f9ea53bf35358693b5eaa4daebdcc8c0"
# HF revision pin for BAAI/bge-small-en-v1.5 (used by transformers/MLX paths)
export HF_BGE_REVISION="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

# Disk-space sanity check (need ~2.5GB free for venv + models + cache)
df_avail_gb=$(df -g . | awk 'NR==2 {print $4}')
if [ "$df_avail_gb" -lt 3 ]; then
  echo "WARN: only ${df_avail_gb}GB free on this volume; recommend 3GB+." >&2
fi

# 1. Python env — prefer 3.12 then 3.11 (coremltools 9 has a crash on 3.13)
PYBIN=""
for cand in python3.12 python3.11 python3.10; do
  if command -v "$cand" >/dev/null 2>&1; then PYBIN="$cand"; break; fi
done
if [ -z "$PYBIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYBIN="python3"
    pyminor=$($PYBIN -c 'import sys; print(sys.version_info.minor)')
    if [ "$pyminor" -ge 13 ]; then
      echo "ERROR: Python 3.${pyminor} detected, but torch 2.7.0 (required for"
      echo "       coremltools 9.0 conversion) has no wheel for Python 3.13/3.14."
      echo "       Install Python 3.12 and re-run:"
      echo "         brew install python@3.12"
      echo "       (Apple's CoreML destructor race on Py3.13+ is a separate issue;"
      echo "       the bench scripts work around it via os._exit(0) once 3.12 is in"
      echo "       use.)"
      exit 1
    fi
  fi
fi
[ -z "$PYBIN" ] && { echo "ERROR: no usable python3 found. Install: brew install python@3.12" >&2; exit 1; }
echo "==> python: $PYBIN ($($PYBIN -c 'import sys; print(sys.version.split()[0])'))"

# Recreate venv if requirements changed (idempotency)
REQ_HASH=$(shasum -a 256 requirements.txt | awk '{print $1}')
if [ -d venv ] && [ -f venv/.requirements.sha256 ] && \
   [ "$(cat venv/.requirements.sha256)" != "$REQ_HASH" ]; then
  echo "==> requirements.txt changed; rebuilding venv"
  rm -rf venv
fi
if [ ! -d venv ]; then
  echo "==> creating venv"
  "$PYBIN" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --quiet --upgrade pip
# Order matters: --no-deps for mlx-embeddings + mlx-vlm to break the
# transformers-5+ transitive constraint, then explicit runtime deps from
# requirements.txt, then anything mlx-embeddings imports at module load.
pip install --quiet --no-deps mlx-embeddings==0.1.0 mlx-vlm==0.4.4 mlx-lm mlx-audio
pip install --quiet -r requirements.txt
# Runtime deps that mlx-vlm/mlx-audio import at module load. Pulled with default
# deps so their own transitive needs (sympy, networkx, etc.) come along.
pip install --quiet Pillow fastapi opencv-python miniaudio llguidance uvicorn datasets
echo "$REQ_HASH" > venv/.requirements.sha256

# 2. Build corpus (deterministic; ~1 sec)
if [ ! -f corpus/corpus_buckets.json ]; then
  echo "==> building corpus"
  python3 corpus/build_corpus.py
fi

# 3. Acquire CoreML mlpackage — prefer release asset over local conversion,
# because fresh conversion via coremltools 9 + torch 2.7 is not reproducible
# across Apple silicon generations (failed on M4 Pro with TypeError; works on M5).
MLPKG_DIR=models/bge-small-en-v1.5.mlpackage
if [ ! -d "$MLPKG_DIR" ]; then
  if [ "${REBUILD_MLPACKAGE:-0}" = "1" ]; then
    echo "==> REBUILD_MLPACKAGE=1 — converting locally (may fail on non-M5 silicon)"
    mkdir -p models
    python3 bench/convert_bge_coreml.py
  else
    echo "==> downloading pre-built mlpackage from release"
    mkdir -p models
    tarball=models/bge-small-en-v1.5.mlpackage.tar.gz
    curl -L --fail -o "$tarball" \
      "https://github.com/${GH_REPO}/releases/download/v0.1.0/bge-small-en-v1.5.mlpackage.tar.gz" \
      || { echo "ERROR: mlpackage download failed."; echo "  Try REBUILD_MLPACKAGE=1 ./bench.sh, or report the issue."; exit 1; }
    got=$(shasum -a 256 "$tarball" | awk '{print $1}')
    if [ "$got" != "$MLPACKAGE_TARBALL_SHA256" ]; then
      echo "ERROR: mlpackage SHA256 mismatch."
      echo "  expected: $MLPACKAGE_TARBALL_SHA256"
      echo "  got:      $got"
      rm -f "$tarball"
      exit 1
    fi
    tar xzf "$tarball" -C models
    rm "$tarball"
  fi
fi

# 4. Download GGUF (one-time, ~67 MB) with SHA256 verification
GGUF=models/bge-small-en-v1.5-f16.gguf
if [ ! -f "$GGUF" ]; then
  echo "==> downloading bge-small GGUF"
  curl -L --fail -o "${GGUF}.tmp" \
    "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f16.gguf"
  got=$(shasum -a 256 "${GGUF}.tmp" | awk '{print $1}')
  if [ "$got" != "$GGUF_SHA256" ]; then
    echo "ERROR: GGUF SHA256 mismatch."
    echo "  expected: $GGUF_SHA256"
    echo "  got:      $got"
    rm -f "${GGUF}.tmp"
    exit 1
  fi
  mv "${GGUF}.tmp" "$GGUF"
fi

mkdir -p results

run_backend() {
  local name="$1"; shift
  echo "==> $name"
  if "$@"; then
    set_status "$name" "ok"
  else
    rc=$?
    set_status "$name" "failed(rc=$rc)"
    echo "WARN: $name backend failed (rc=$rc). Result block will show — for this row." >&2
  fi
}

# 5. Run device-placement probes (separate process to isolate crash class).
# These produce results/devices_<unit>.json that bench_coreml.py picks up.
echo "==> device-placement probes"
for u in ane gpu cpu; do
  python3 bench/probe_devices.py --compute "$u" --out "results/devices_${u}.json" \
    || echo "WARN: device probe for $u failed (informational; bench will still run)." >&2
done

# 6. Run benches
run_backend "CoreML ANE"  python3 bench/bench_coreml.py --compute ane --out results/coreml_ane.json
run_backend "CoreML GPU"  python3 bench/bench_coreml.py --compute gpu --out results/coreml_gpu.json
run_backend "CoreML CPU"  python3 bench/bench_coreml.py --compute cpu --out results/coreml_cpu.json
if command -v llama-embedding >/dev/null 2>&1; then
  run_backend "llama.cpp Metal" bench/bench_llama_v2.sh
else
  set_status "llama.cpp Metal" "skipped (llama.cpp not installed; brew install llama.cpp)"
  echo "WARN: llama-embedding not found. Install with: brew install llama.cpp" >&2
fi
run_backend "MLX-embeddings" python3 bench/bench_mlx.py

# 7. Aggregate + print pasteable result block
echo
echo "================================================================"
echo "RESULT BLOCK (copy everything between BEGIN/END into your issue)"
echo "================================================================"

# Pass backend statuses to Python via a JSON blob
status_json=$(get_statuses_json)

# Write result block to both stdout AND a file so submitters can attach it
RESULT_MD="results/RESULT.md"
{
  echo "===BEGIN RESULT==="
  python3 - "$status_json" <<'PY'
import json, os, subprocess, sys
from pathlib import Path

statuses = json.loads(sys.argv[1])
R = Path("results")
def load(name):
    p = R / name
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except Exception as e: return {"_parse_error": f"{type(e).__name__}: {e}"}

results = {
    "coreml_ane": load("coreml_ane.json"),
    "coreml_gpu": load("coreml_gpu.json"),
    "coreml_cpu": load("coreml_cpu.json"),
    "llama_metal": load("llama_metal.json"),
    "mlx": load("mlx_embeddings.json"),
    "devices_ane": load("devices_ane.json"),
    "devices_gpu": load("devices_gpu.json"),
    "devices_cpu": load("devices_cpu.json"),
}

def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()
chip = sh("sysctl","-n","machdep.cpu.brand_string")
mem_gb = round(int(sh("sysctl","-n","hw.memsize")) / 1024**3)
macos = sh("sw_vers","-productVersion"); build = sh("sw_vers","-buildVersion")
def llama_v():
    for stream in ("stdout","stderr"):
        try:
            r = subprocess.run(["llama-embedding","--version"], capture_output=True, text=True)
            s = getattr(r, stream)
            for line in (s or "").split("\n"):
                if line.strip(): return line.strip()
        except Exception: pass
    return "n/a"

print(f"## bench result")
print(f"chip: {chip}")
print(f"memory: {mem_gb} GB")
print(f"macOS: {macos} ({build})")
print(f"python: {sys.version.split()[0]}")
print(f"llama.cpp: {llama_v()}")
print()
print("**Backend status:**")
for name in ("CoreML ANE","CoreML GPU","CoreML CPU","llama.cpp Metal","MLX-embeddings"):
    print(f"- {name}: {statuses.get(name, 'unknown')}")
print()
print("| backend | short b=1 | short batched | medium b=1 | medium batched | long b=1 | long batched | cold (s) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")

def b1(d, b):
    try: return f"{d['buckets'][b]['batch1_sent_per_s']:.1f}"
    except (KeyError, TypeError): return "—"
def bN(d, b):
    try: return f"{d['buckets'][b]['batchN_sent_per_s']:.1f}"
    except (KeyError, TypeError): return "—"
def cold(d):
    try: return f"{d['cold_start_s']:.2f}"
    except (KeyError, TypeError): return "—"
def llama_total(d, b):
    try:
        runs = [r for r in d['buckets'][b]['runs'] if r.get('status') == 'ok']
        if len(runs) < 2: return "—"  # need at least 2 successful runs to drop run 1
        vals = [r['sent_per_s_total'] for r in runs[1:]]
        return f"{sum(vals)/len(vals):.1f}"
    except (KeyError, TypeError): return "—"

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

# Device-placement summary (the methodology-credibility line)
print()
print("**Device placement (verified, not hint):**")
for u, key in [("ANE","devices_ane"),("GPU","devices_gpu"),("CPU","devices_cpu")]:
    d = results[key]
    if d and d.get("device_placement_available"):
        ops = d.get("op_count_by_device", {})
        if ops:
            print(f"- {u} requested: {ops}")
    elif d:
        print(f"- {u} probe error: {d.get('error','unknown')}")
PY
  echo "===END RESULT==="
} | tee "$RESULT_MD"

echo
echo "Result block also saved to: $RESULT_MD"
echo "Submit at: https://github.com/${GH_REPO}/issues/new?template=bench-result.md"
