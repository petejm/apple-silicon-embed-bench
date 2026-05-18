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

echo '{ "model": "bge-small-en-v1.5-f16.gguf", "backend": "llama.cpp Metal (internal timing)", "note": "llama.cpp BERT-embed batches sentences (n_seq=66 typical) into one forward pass; total_time / n_sentences = realistic throughput", "buckets": {' > "$OUT.tmp"

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
    max_n_seq=$(grep "batch_decode: n_tokens" "$log" | sed -nE 's/.*n_seq = ([0-9]+).*/\1/p' | sort -n | tail -1 || true)

    if [ -z "$prompt_eval_ms" ] || [ -z "$prompt_tokens" ] || [ -z "$total_ms" ]; then
      echo "[llama-v2/$bucket/$r] FAILED — could not parse perf output. Last log lines:" >&2
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
