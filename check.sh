#!/usr/bin/env bash
# Pre-flight check for apple-silicon-embed-bench. Run before ./bench.sh to
# surface environment issues up front rather than ~5 minutes into a bench.
#
# Usage: ./check.sh
# Exits 0 if everything is ready, 1 if blockers found.
# bench.sh runs this automatically on startup.

set -uo pipefail

# Colors / glyphs work even on basic terminals
OK=" ✓ "
WARN=" ⚠ "
ERR=" ✗ "

ERRORS=0
WARNINGS=0

note() { echo "$1"; }
ok()   { echo "${OK}$1"; }
warn() { echo "${WARN}$1"; WARNINGS=$((WARNINGS+1)); }
err()  { echo "${ERR}$1"; ERRORS=$((ERRORS+1)); }

echo "==> apple-silicon-embed-bench — pre-flight check"
echo

# 1. OS check
echo "[CHECK] Operating system"
os=$(uname -s)
if [ "$os" != "Darwin" ]; then
  err "macOS required (this is $os). The bench tests CoreML, MLX, and Metal — Apple-only."
else
  os_ver=$(sw_vers -productVersion)
  major=${os_ver%%.*}
  if [ "$major" -lt 14 ]; then
    err "macOS $os_ver detected; need 14 (Sonoma) or later."
    note "      Upgrade macOS, then re-run."
  else
    ok "macOS $os_ver (build $(sw_vers -buildVersion))"
  fi
fi
echo

# 2. CPU architecture
echo "[CHECK] CPU architecture"
arch=$(uname -m)
if [ "$arch" != "arm64" ]; then
  err "arm64 required (this is $arch). The bench measures Apple-silicon paths."
  note "      Intel Macs are not supported."
else
  chip=$(sysctl -n machdep.cpu.brand_string)
  ok "arm64 ($chip)"
fi
echo

# 2b. Rosetta sanity — ensure Python (if installed) isn't running under Rosetta
if command -v python3 >/dev/null 2>&1; then
  py_arch=$(python3 -c 'import platform; print(platform.machine())' 2>/dev/null || echo "unknown")
  if [ "$py_arch" != "arm64" ] && [ "$arch" = "arm64" ]; then
    warn "python3 is running under $py_arch (likely Rosetta). Performance will be"
    note "      severely degraded. Install a native arm64 Python: brew install python@3.12"
  fi
fi

# 3. Memory
echo "[CHECK] Memory"
mem_gb=$(python3 -c "print(round($(sysctl -n hw.memsize) / 1024**3))" 2>/dev/null || echo "0")
if [ "$mem_gb" -lt 8 ]; then
  warn "${mem_gb}GB detected. The bench needs ~2GB free for venv + models + cache."
  note "      Should still run; MLX 100-batched long bucket may OOM on 8GB."
else
  ok "${mem_gb}GB unified memory"
fi
echo

# 4. Disk space
echo "[CHECK] Disk space (need ~3GB free)"
df_avail_gb=$(df -g . | awk 'NR==2 {print $4}')
if [ "$df_avail_gb" -lt 3 ]; then
  err "${df_avail_gb}GB free on the current volume; need ~3GB for venv + GGUF + mlpackage + cache."
  note "      Free up space or run from a volume with more space."
else
  ok "${df_avail_gb}GB free"
fi
echo

# 5. Homebrew (for installing prereqs if needed)
echo "[CHECK] Homebrew"
if command -v brew >/dev/null 2>&1; then
  ok "brew at $(which brew)"
elif [ -x /opt/homebrew/bin/brew ]; then
  warn "brew installed but not on PATH. Run:  eval \"\$(/opt/homebrew/bin/brew shellenv)\""
elif [ -x /usr/local/bin/brew ]; then
  warn "brew installed but not on PATH. Run:  eval \"\$(/usr/local/bin/brew shellenv)\""
else
  warn "brew not found. Some installation instructions below assume brew is available."
  note "      Install: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
fi
echo

# 6. Python version & availability
echo "[CHECK] Python (need 3.10–3.13; 3.14+ has no torch 2.7 wheel; <=3.9 too old for coremltools 9)"
PYBIN=""
for cand in python3.12 python3.11 python3.10 python3.13; do
  if command -v "$cand" >/dev/null 2>&1; then PYBIN="$cand"; break; fi
done
if [ -z "$PYBIN" ] && command -v python3 >/dev/null 2>&1; then
  PYBIN="python3"
fi

if [ -z "$PYBIN" ]; then
  err "No python3 found."
  note "      Install: brew install python@3.12"
else
  pyver=$($PYBIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
  pyminor=$($PYBIN -c 'import sys; print(sys.version_info.minor)')
  if [ "$pyminor" -ge 14 ]; then
    err "Python $pyver detected. The bench stack is not viable on 3.14:"
    note "      - torch 2.7.0 has no cp314 wheel (and torch 2.9+ has a typing.Union"
    note "        AttributeError on 3.14 in torch.ao.quantization)"
    note "      - coremltools 9.0 has no cp314 wheel (sdist fallback installs but"
    note "        libcoremlpython.so is missing; all MLModel calls fail at runtime)"
    note "      Install: brew install python@3.12"
  elif [ "$pyminor" -lt 10 ]; then
    err "Python $pyver too old for coremltools 9 / torch 2.7."
    note "      Install: brew install python@3.12"
  elif [ "$pyminor" -eq 13 ]; then
    warn "Python $pyver (will work, but triggers Apple's CoreML destructor race;"
    note "      bench works around it via os._exit(0). For a clean stack: brew install python@3.12)"
  else
    ok "Python $pyver ($PYBIN)"
  fi
fi
echo

# 7. llama.cpp (optional but needed for the llama.cpp row)
echo "[CHECK] llama.cpp (optional; one of 5 backends)"
if command -v llama-embedding >/dev/null 2>&1; then
  ok "llama-embedding at $(which llama-embedding)"
  # Check if the brew build has tensor units enabled
  tensor_state=$(llama-embedding --help 2>&1 | grep "has tensor" | head -1 || echo "")
  if [ -z "$tensor_state" ]; then
    # Probe via a quick run — but skip if it'd take too long. Just note it.
    note "      To check Apple tensor-unit state on M5/newer: look for 'has tensor = true/false'"
    note "      in the init log when llama-embedding runs."
  fi
else
  warn "llama-embedding not found. The bench will skip the llama.cpp row."
  note "      To include it: brew install llama.cpp"
fi
echo

# 8. Network reachability (the bench downloads ~70MB from GitHub + HuggingFace)
echo "[CHECK] Network reachability"
for host in github.com huggingface.co; do
  if curl -sSf --max-time 5 --head "https://$host/" >/dev/null 2>&1; then
    ok "$host reachable"
  else
    err "$host not reachable. The bench downloads model artifacts from here."
    note "      Check your network. Bench cannot complete on first run without these."
  fi
done
echo

# 9. Xcode CLI tools (for native arm64 wheels that need a C compiler)
echo "[CHECK] Xcode CLI tools (for any pip-source builds)"
if xcode-select -p >/dev/null 2>&1; then
  xcl_path=$(xcode-select -p)
  ok "xcode-select at $xcl_path"
else
  warn "Xcode CLI tools not installed. Most deps have arm64 wheels; you'll only need"
  note "      these if a future dep is source-only. Install: xcode-select --install"
fi
echo

# 10. Existing venv state (informational)
if [ -d venv ]; then
  echo "[CHECK] Existing venv"
  if [ -f venv/.requirements.sha256 ]; then
    cur=$(shasum -a 256 requirements.txt 2>/dev/null | awk '{print $1}')
    saved=$(cat venv/.requirements.sha256 2>/dev/null)
    if [ "$cur" = "$saved" ]; then
      ok "venv up-to-date with current requirements.txt"
    else
      warn "venv was built against different requirements.txt — bench.sh will rebuild it"
    fi
  else
    warn "venv exists but has no requirements.sha256 stamp — bench.sh will rebuild it"
  fi
  echo
fi

# Summary
echo "================================================================"
if [ "$ERRORS" -gt 0 ]; then
  echo "$ERR $ERRORS blocker(s), $WARNINGS warning(s). Fix the blockers, then run ./bench.sh."
  exit 1
elif [ "$WARNINGS" -gt 0 ]; then
  echo "$WARN $WARNINGS warning(s). bench.sh should still run; expect the noted limitations."
  echo "       Run ./bench.sh when ready."
  exit 0
else
  echo "$OK All checks passed. Run ./bench.sh to start the benchmark."
  exit 0
fi
