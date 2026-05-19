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

# Run pre-flight check unless explicitly skipped (SKIP_CHECK=1)
if [ "${SKIP_CHECK:-0}" != "1" ] && [ -x "$ROOT/check.sh" ]; then
  if ! "$ROOT/check.sh"; then
    echo
    echo "Pre-flight check found blockers. Fix them and re-run ./bench.sh"
    echo "(or set SKIP_CHECK=1 to bypass at your own risk)."
    exit 1
  fi
  echo
fi

echo "==> apple-silicon-embed-bench"
echo "==> repo:  $ROOT"
echo "==> macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
echo "==> chip:  $(sysctl -n machdep.cpu.brand_string)"
# (memory printed below, after Python interpreter selection — we use $PYBIN
# for the round() so we don't take a hard dependency on `python3` being on
# PATH before we know we have a usable Python at all.)
echo

# Track per-backend status so the result block can flag failures explicitly.
# macOS ships bash 3.2 which lacks associative arrays; use a temp status file.
STATUS_FILE="$(mktemp)"
# Cleanup on any termination path — EXIT covers normal exit; INT/TERM cover
# Ctrl-C and external kill. The tarball/mlpackage download path can leave
# *.partial.* staging dirs; sweep them too.
cleanup() {
  rm -f "$STATUS_FILE"
  rm -rf models/*.partial.* 2>/dev/null || true
}
trap cleanup EXIT INT TERM
set_status() { printf "%s\t%s\n" "$1" "$2" >> "$STATUS_FILE"; }
get_statuses_json() {
  "$PYBIN" -c '
import json, sys
d = {}
for line in open(sys.argv[1]):
    k, _, v = line.rstrip("\n").partition("\t")
    if k: d[k] = v
print(json.dumps(d))
' "$STATUS_FILE"
}

# Canonical upstream — where the pre-built model artifact lives. Always
# hits petejm/apple-silicon-embed-bench because forks don't republish the
# .mlpackage release asset.
UPSTREAM_REPO="petejm/apple-silicon-embed-bench"

# Submission URL (printed at the end). Forks want their own issues, so
# compute this from the local git remote. Falls back to upstream if not a
# GitHub remote.
REMOTE_URL=$(git config --get remote.origin.url 2>/dev/null || echo "")
case "$REMOTE_URL" in
  git@github.com:*) GH_REPO="${REMOTE_URL#git@github.com:}"; GH_REPO="${GH_REPO%.git}" ;;
  https://github.com/*) GH_REPO="${REMOTE_URL#https://github.com/}"; GH_REPO="${GH_REPO%.git}" ;;
  *) GH_REPO="$UPSTREAM_REPO" ;;
esac

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

# 1. Python env — prefer 3.12 (clean), then 3.11, 3.10, 3.13. Hard-fail on 3.14+
# (torch 2.7.0 has no Apple-silicon wheel) and on <=3.9 (too old for our deps).
# 3.13 works but triggers Apple's coremltools destructor race in MLE5ExecutionStream;
# the bench scripts work around it via os._exit(0). 3.12 is the cleanest target.
# Selection logic is shared with check.sh via lib/python.sh.
# shellcheck disable=SC1091
. "$ROOT/lib/python.sh"
pick_python || true
[ -z "$PYBIN" ] && { echo "ERROR: no usable python3 found. Install: brew install python@3.12" >&2; exit 1; }

pyminor=$($PYBIN -c 'import sys; print(sys.version_info.minor)')
if [ "$pyminor" -ge 14 ]; then
  echo "ERROR: Python 3.${pyminor} detected. The bench stack is not viable on 3.14+:"
  echo "       - torch 2.7.0 has no cp314 wheel; torch 2.9+ throws AttributeError"
  echo "         on Py3.14 in torch.ao.quantization (typing.Union semantics changed)"
  echo "       - coremltools 9.0 sdist installs but libcoremlpython.so is not built"
  echo "         for 3.14; MLModel calls fail at runtime"
  echo "       Install Python 3.12 and re-run:"
  echo "         brew install python@3.12"
  exit 1
fi
if [ "$pyminor" -lt 10 ]; then
  echo "ERROR: Python 3.${pyminor} is too old for coremltools 9 / torch 2.7."
  echo "       Install Python 3.12 and re-run:"
  echo "         brew install python@3.12"
  exit 1
fi
if [ "$pyminor" -eq 13 ]; then
  echo "INFO: Python 3.13 detected. The bench will work via the os._exit(0)"
  echo "      workaround for Apple's CoreML destructor race. For a cleaner"
  echo "      stack consider Python 3.12 (brew install python@3.12)."
fi
echo "==> python: $PYBIN ($($PYBIN -c 'import sys; print(sys.version.split()[0])'))"

# Memory print uses $PYBIN (proper rounding so 24GB doesn't print as 23GB).
mem_gb=$($PYBIN -c "print(round($(sysctl -n hw.memsize) / 1024**3))")
echo "==> mem:   ${mem_gb} GB"

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
# After venv activation, repoint PYBIN at the venv's python so every
# subsequent "$PYBIN" invocation picks up the venv's installed deps. (Before
# this point, PYBIN was the system-resolved interpreter we used to BUILD the
# venv.) This makes the bench self-consistent — no bare `python3` calls that
# could resolve to a different interpreter than the one with our deps.
PYBIN="$(command -v python)"
pip install --quiet --upgrade pip
# Order matters: --no-deps for mlx-embeddings + mlx-vlm to break the
# transformers-5+ transitive constraint, then explicit runtime deps from
# requirements.txt, then anything mlx-embeddings imports at module load.
pip install --quiet --no-deps mlx-embeddings==0.1.0 mlx-vlm==0.4.4 mlx-lm mlx-audio
pip install --quiet -r requirements.txt
# Runtime deps that mlx-vlm/mlx-audio import at module load are now pinned
# inside requirements.txt (Pillow, fastapi, opencv-python, miniaudio,
# llguidance, uvicorn, datasets). Previously these were unpinned `pip
# install` calls here, which produced silent skew across community
# submissions — pinning them in requirements.txt makes the venv-rebuild
# guard (REQ_HASH) catch upgrades.
echo "$REQ_HASH" > venv/.requirements.sha256

# 2. Build corpus (deterministic; ~1 sec)
if [ ! -f corpus/corpus_buckets.json ]; then
  echo "==> building corpus"
  "$PYBIN" corpus/build_corpus.py
fi

# 3. Acquire CoreML mlpackage — prefer release asset over local conversion,
# because fresh conversion via coremltools 9 + torch 2.7 is not reproducible
# across Apple silicon generations (failed on M4 Pro with TypeError; works on M5).
MLPKG_DIR=models/bge-small-en-v1.5.mlpackage
if [ ! -d "$MLPKG_DIR" ]; then
  if [ "${REBUILD_MLPACKAGE:-0}" = "1" ]; then
    echo "==> REBUILD_MLPACKAGE=1 — converting locally (may fail on non-M5 silicon)"
    mkdir -p models
    "$PYBIN" bench/convert_bge_coreml.py
  else
    echo "==> downloading pre-built mlpackage from release"
    mkdir -p models
    tarball=models/bge-small-en-v1.5.mlpackage.tar.gz
    curl -L --fail -o "$tarball" \
      "https://github.com/${UPSTREAM_REPO}/releases/download/v1.0.0/bge-small-en-v1.5.mlpackage.tar.gz" \
      || { echo "ERROR: mlpackage download failed."; echo "  Try REBUILD_MLPACKAGE=1 ./bench.sh, or report the issue."; exit 1; }
    got=$(shasum -a 256 "$tarball" | awk '{print $1}')
    if [ "$got" != "$MLPACKAGE_TARBALL_SHA256" ]; then
      echo "ERROR: mlpackage SHA256 mismatch."
      echo "  expected: $MLPACKAGE_TARBALL_SHA256"
      echo "  got:      $got"
      rm -f "$tarball"
      exit 1
    fi
    # Atomic extract: extract to a staging dir, then mv into place on success.
    # An interrupted `tar xzf -C models` otherwise leaves a partial
    # models/bge-small-en-v1.5.mlpackage/ dir that the next run's
    # `[ ! -d ... ]` gate would silently skip re-downloading.
    extract_dir=$(mktemp -d "${MLPKG_DIR}.partial.XXXXXX")
    if ! tar xzf "$tarball" -C "$extract_dir"; then
      echo "ERROR: mlpackage tarball extract failed; not leaving a partial directory in place."
      rm -rf "$extract_dir"
      rm -f "$tarball"
      exit 1
    fi
    if [ ! -d "$extract_dir/bge-small-en-v1.5.mlpackage" ]; then
      echo "ERROR: tarball did not contain expected bge-small-en-v1.5.mlpackage/ directory."
      rm -rf "$extract_dir"
      rm -f "$tarball"
      exit 1
    fi
    mv "$extract_dir/bge-small-en-v1.5.mlpackage" "$MLPKG_DIR"
    rmdir "$extract_dir" 2>/dev/null || rm -rf "$extract_dir"
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

# 4b. Cross-backend parity check (TR-3 D1/D3).
# Runs before any per-backend bench so we abort early if the three backends
# don't agree on the same function. results/parity.json gets surfaced in the
# RESULT block.
echo "==> cross-backend parity check (CoreML vs MLX vs llama)"
if "$PYBIN" bench/verify_parity.py --allow-missing-backend; then
  set_status "parity" "ok"
else
  parity_rc=$?
  set_status "parity" "failed(rc=$parity_rc)"
  echo "WARN: parity check failed (rc=$parity_rc). Per-backend numbers will still be" >&2
  echo "      collected, but cross-backend ratio claims are not justified until parity" >&2
  echo "      is restored. See results/parity.json for diagnostics." >&2
fi
echo

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
# Track probe status — if probes silently fail, the methodology
# selling-point ("verified device placement") is broken without anyone
# noticing. Surface them in the result block.
echo "==> device-placement probes"
for u in ane gpu cpu; do
  if "$PYBIN" bench/probe_devices.py --compute "$u" --out "results/devices_${u}.json"; then
    set_status "probe $u" "ok"
  else
    rc=$?
    set_status "probe $u" "failed(rc=$rc)"
    echo "WARN: device probe for $u failed (rc=$rc). Bench will still run, but device-placement claims are unverified for this row." >&2
  fi
done

# 6. Run benches
run_backend "CoreML ANE"  "$PYBIN" bench/bench_coreml.py --compute ane --out results/coreml_ane.json
run_backend "CoreML GPU"  "$PYBIN" bench/bench_coreml.py --compute gpu --out results/coreml_gpu.json
run_backend "CoreML CPU"  "$PYBIN" bench/bench_coreml.py --compute cpu --out results/coreml_cpu.json
if command -v llama-embedding >/dev/null 2>&1; then
  run_backend "llama.cpp Metal" bench/bench_llama_v2.sh
else
  set_status "llama.cpp Metal" "skipped (llama.cpp not installed; brew install llama.cpp)"
  echo "WARN: llama-embedding not found. Install with: brew install llama.cpp" >&2
fi
run_backend "MLX-embeddings" "$PYBIN" bench/bench_mlx.py

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
  "$PYBIN" - "$status_json" <<'PY'
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
print("**Device-placement probe status:**")
for name in ("probe ane","probe gpu","probe cpu"):
    print(f"- {name}: {statuses.get(name, 'unknown')}")
print()
# Backend parity (cross-backend cosine + token-count histogram)
print("**Backend parity (cross-backend cosine + token-count histogram):**")
print(f"- parity: {statuses.get('parity', 'unknown')}")
parity_path = R / "parity.json"
if parity_path.exists():
    try:
        p = json.loads(parity_path.read_text())
        cos = p.get("checks", {}).get("cosine", {}).get("pairs", {})
        for pair, info in cos.items():
            if info.get("ok"):
                print(f"  - cosine {pair}: min={info.get('min', 0):.4f}, mean={info.get('mean', 0):.4f} (>= {p.get('cosine_threshold')})")
            else:
                err = info.get("error") or f"min={info.get('min', 0):.4f} < {p.get('cosine_threshold')}"
                print(f"  - cosine {pair}: FAIL — {err}")
        tc = p.get("checks", {}).get("token_counts", {})
        if tc:
            ref = tc.get("ref_backend", "?")
            for name, ok in tc.get("matches", {}).items():
                print(f"  - token-count {ref} vs {name}: {'ok' if ok else 'FAIL'}")
    except Exception as e:
        print(f"  - (could not read parity.json: {type(e).__name__}: {e})")
print()
print("| backend | short b=1 | short batched | medium b=1 | medium batched | long b=1 | long batched | cold (s) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")

def _fmt_stats(runs):
    """Format mean / range / sigma for a list of per-run throughputs.
    Drops first run (warmup), like the bench protocol elsewhere.
    Returns e.g. `565.2 ±2.3 [560…568]`. Single line per cell.
    """
    if not runs or len(runs) < 2:
        # With only the warmup or fewer, show whatever we have.
        if runs:
            return f"{runs[0]:.1f}"
        return "—"
    vals = runs[1:]  # drop run 1
    mean = sum(vals) / len(vals)
    lo, hi = min(vals), max(vals)
    if len(vals) > 1:
        var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
        sigma = var ** 0.5
        return f"{mean:.1f} ±{sigma:.1f} [{lo:.0f}…{hi:.0f}]"
    return f"{mean:.1f} [{lo:.0f}…{hi:.0f}]"

def b1(d, b):
    try:
        runs = d['buckets'][b].get('batch1_sent_per_s_runs')
        if runs: return _fmt_stats(runs)
        return f"{d['buckets'][b]['batch1_sent_per_s']:.1f}"
    except (KeyError, TypeError): return "—"
def bN(d, b):
    try:
        runs = d['buckets'][b].get('batchN_sent_per_s_runs')
        if runs: return _fmt_stats(runs)
        return f"{d['buckets'][b]['batchN_sent_per_s']:.1f}"
    except (KeyError, TypeError): return "—"
def cold(d):
    try: return f"{d['cold_start_s']:.2f}"
    except (KeyError, TypeError): return "—"
def llama_total(d, b):
    try:
        # Default-accept untagged runs: older bench_llama_v2.sh outputs (and
        # the M4 Pro community results, pre-status-backfill) didn't emit a
        # status field. Tagged runs default to 'ok'; only explicit failures
        # ('failed', 'parse_failed') are filtered out.
        runs = [r for r in d['buckets'][b]['runs'] if r.get('status', 'ok') == 'ok']
        if len(runs) < 2: return "—"  # need at least 2 successful runs to drop run 1
        vals = [r['sent_per_s_total'] for r in runs]
        # _fmt_stats drops run 1 internally.
        return _fmt_stats(vals)
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
