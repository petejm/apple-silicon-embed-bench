#!/usr/bin/env bash
# llama.cpp Metal bench v2 — use llama's internal timings (not wall clock),
# because process spawn dominates wall and isn't part of inference cost.
# llama.cpp batches sentences internally (n_seq=66 typical), which is the
# realistic deployment shape.
#
# Hardened against silent failure: if log parsing returns empty, the bucket
# is marked status="failed" rather than writing zeros that look like real
# measurements. The aggregator surfaces these explicitly.
set -euo pipefail
# Defang locale-sensitive number formatting (e.g. decimal comma in some locales)
export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$ROOT/models/bge-small-en-v1.5-f16.gguf"
OUT="$ROOT/results/llama_metal.json"
mkdir -p "$ROOT/results"
# Clean up the partial JSON if anything below aborts.
trap 'rm -f "$OUT.tmp"' EXIT

# Pin a reasonable batch size; default varies by brew build.
LLAMA_ARGS="--pooling mean --embd-output-format array --embd-normalize 2 -ngl 99 -b 4096 -ub 4096"

if [ ! -f "$MODEL" ]; then
  echo "FAIL: $MODEL not found. Did bench.sh download fail?" >&2
  echo '{"status":"failed","reason":"GGUF missing","model":"bge-small-en-v1.5-f16.gguf","buckets":{}}' > "$OUT"
  exit 1
fi

# Probe the GGUF dtype tag once — surface it in the result block so the
# three-backend dtype comparison (MLX float16, CoreML compute_precision,
# llama gguf_dtype) is auditable from the artifacts alone.
gguf_dtype="unknown"
if command -v gguf-dump >/dev/null 2>&1; then
  gguf_dtype=$(gguf-dump --no-tensors "$MODEL" 2>/dev/null | grep -i "general.file_type\|general\.quantization_version\|tensor_data_layout" | head -3 | tr '\n' ';' || echo "unknown")
fi
if [ "$gguf_dtype" = "unknown" ]; then
  # Fallback: parse the early init log of a tiny dry run for 'f16' / 'q8' / 'f32' markers.
  init_log=$(mktemp)
  echo "test" > "${init_log}.txt"
  # shellcheck disable=SC2086
  llama-embedding -m "$MODEL" -f "${init_log}.txt" $LLAMA_ARGS > /dev/null 2> "$init_log" || true
  gguf_dtype=$(grep -oE "ftype +=? +[A-Za-z0-9_-]+|file type:[^,]*|all F16|all F32|all Q[0-9]_[0-9KS]+" "$init_log" 2>/dev/null | head -3 | tr '\n' ';' || echo "unknown")
  [ -z "$gguf_dtype" ] && gguf_dtype="unparsed"
  rm -f "$init_log" "${init_log}.txt"
fi
echo "[llama-v2] gguf_dtype: $gguf_dtype" >&2

# JSON-escape the dtype string (replace quotes/backslashes with safe chars).
gguf_dtype_esc=$(printf '%s' "$gguf_dtype" | sed 's/\\/\\\\/g; s/"/\\"/g')

echo "{ \"model\": \"bge-small-en-v1.5-f16.gguf\", \"backend\": \"llama.cpp Metal (internal timing)\", \"gguf_dtype\": \"$gguf_dtype_esc\", \"note\": \"llama.cpp BERT-embed batches sentences (n_seq=66 typical) into one forward pass; total_time / n_sentences = realistic throughput\", \"buckets\": {" > "$OUT.tmp"

first=1
overall_ok=1
for bucket in short medium long; do
  CORPUS="$ROOT/corpus/corpus_llama_${bucket}.txt"
  runs_json=""
  bucket_ok=1
  for r in 1 2 3 4 5; do
    log=$(mktemp)
    llama_rc=0
    # shellcheck disable=SC2086
    llama-embedding -m "$MODEL" -f "$CORPUS" $LLAMA_ARGS > /dev/null 2> "$log" || llama_rc=$?

    if [ "$llama_rc" -ne 0 ]; then
      echo "[llama-v2/$bucket/$r] FAILED rc=$llama_rc — last log lines:" >&2
      tail -5 "$log" >&2
      runs_json+="{\"run\":$r,\"status\":\"failed\",\"llama_rc\":$llama_rc},"
      bucket_ok=0
      rm -f "$log"
      continue
    fi

    # Parse llama_perf_context_print lines. Be defensive — if any field is
    # missing, treat the run as failed and surface it (not silent zeros).
    # Pipelines fronted by grep must be `|| true` under `set -euo pipefail`,
    # otherwise a no-match grep aborts the whole script before the explicit
    # empty-check below ever runs (silent-fail = the bug this script was
    # written to prevent). Regex tolerates both `1234.5 ms` and `1234 ms`.
    prompt_eval_ms=$(grep "prompt eval time" "$log" | head -1 | sed -nE 's/.*= +([0-9]+(\.[0-9]+)?) ms \/.*/\1/p' || true)
    prompt_tokens=$(grep "prompt eval time" "$log" | head -1 | sed -nE 's/.*ms \/ +([0-9]+) tokens.*/\1/p' || true)
    total_ms=$(grep "total time" "$log" | head -1 | sed -nE 's/.*= +([0-9]+(\.[0-9]+)?) ms \/.*/\1/p' || true)
    n_batches=$(grep -c "batch_decode: n_tokens" "$log" || true)
    # Wrap the whole pipeline so a no-match `grep` doesn't abort under
    # `set -o pipefail` BEFORE `|| true` ever runs (TR-3 B2). Subshell
    # `{...}` keeps stderr redirect and the final `|| echo ""` scoped to
    # the entire chain, not just the last `tail -1`.
    max_n_seq=$({ grep "batch_decode: n_tokens" "$log" | sed -nE 's/.*n_seq = ([0-9]+).*/\1/p' | sort -n | tail -1; } 2>/dev/null || echo "")

    # Reject empty strings AND numeric zero. llama.cpp's "= 0 ms" output for
    # short prompts will legitimately match the grep regex above and would
    # otherwise produce sent_per_s=0 / tokens_per_s=0 shipped as real data.
    # `awk 'BEGIN{exit !(x+0 == 0)}'` returns 0 (success) when x parses as
    # numeric zero, so the `||` chain treats numeric-zero like empty.
    is_zero() { awk -v x="$1" 'BEGIN{exit !(x+0 == 0)}'; }
    if [ -z "$prompt_eval_ms" ] || is_zero "$prompt_eval_ms" \
       || [ -z "$prompt_tokens" ] || is_zero "$prompt_tokens" \
       || [ -z "$total_ms" ] || is_zero "$total_ms"; then
      echo "[llama-v2/$bucket/$r] FAILED — could not parse perf output (empty or numeric zero). Last log lines:" >&2
      tail -10 "$log" >&2
      runs_json+="{\"run\":$r,\"status\":\"parse_failed\"},"
      bucket_ok=0
      rm -f "$log"
      continue
    fi

    # Sane defaults for optional fields
    [ -z "$max_n_seq" ] && max_n_seq=0

    sent_per_s_total=$(python3 -c "t=$total_ms; print(100.0/(t/1000.0) if t>0 else 0)")
    sent_per_s_eval=$(python3 -c "t=$prompt_eval_ms; print(100.0/(t/1000.0) if t>0 else 0)")
    tokens_per_s=$(python3 -c "t=$prompt_eval_ms; n=$prompt_tokens; print(n/(t/1000.0) if t>0 else 0)")

    runs_json+="{\"run\":$r,\"status\":\"ok\",\"prompt_eval_ms\":$prompt_eval_ms,\"prompt_tokens\":$prompt_tokens,\"total_ms\":$total_ms,\"n_batches\":$n_batches,\"max_n_seq\":$max_n_seq,\"sent_per_s_total\":$sent_per_s_total,\"sent_per_s_eval\":$sent_per_s_eval,\"tokens_per_s\":$tokens_per_s},"
    echo "[llama-v2/$bucket/$r] eval=${prompt_eval_ms}ms (${prompt_tokens}tok) total=${total_ms}ms sent/s_total=$sent_per_s_total" >&2
    rm -f "$log"
  done
  runs_json="${runs_json%,}"
  bucket_status="ok"
  [ "$bucket_ok" -eq 0 ] && bucket_status="partial" && overall_ok=0
  if [ "$first" -eq 1 ]; then first=0; else echo "," >> "$OUT.tmp"; fi
  echo "\"$bucket\": { \"status\": \"$bucket_status\", \"runs\": [$runs_json] }" >> "$OUT.tmp"
done
echo "},\"status\":\"$([ "$overall_ok" -eq 1 ] && echo ok || echo partial)\"}" >> "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "wrote $OUT"
