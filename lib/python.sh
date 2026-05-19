# shellcheck shell=bash
# Shared Python interpreter selection. Sourced by both check.sh and bench.sh.
#
# Preference order: 3.12 (cleanest), 3.11, 3.10, 3.13 (works via os._exit
# workaround for Apple's coremltools destructor race). Falls back to
# bare `python3` if none of those are present.
#
# Sets PYBIN to the selected interpreter (or empty string if nothing found).
# Callers should `[ -z "$PYBIN" ]` and error appropriately for their context.
#
# This file deliberately does NOT exit/error itself — sourcing should be
# side-effect-free except for setting PYBIN. The caller decides how strict
# to be (check.sh accumulates warnings; bench.sh hard-fails).

pick_python() {
  local cand
  PYBIN=""
  for cand in python3.12 python3.11 python3.10 python3.13; do
    if command -v "$cand" >/dev/null 2>&1; then
      PYBIN="$cand"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    PYBIN="python3"
    return 0
  fi
  return 1
}
