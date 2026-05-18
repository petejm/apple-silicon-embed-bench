#!/usr/bin/env bash
# llama.cpp Metal bench v2 — use llama's internal timings (not wall clock),
# because process spawn dominates wall and isn't part of inference cost.
# llama.cpp batches sentences internally (n_seq=66 typical), which is the
# realistic deployment shape.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$ROOT/models/bge-small-en-v1.5-f16.gguf"
OUT="$ROOT/results/llama_metal.json"

echo '{ "model": "bge-small-en-v1.5-f16.gguf", "backend": "llama.cpp Metal (internal timing)", "note": "llama.cpp BERT-embed batches sentences (n_seq=66 typical) into one forward pass; total_time / n_sentences = realistic throughput", "buckets": {' > "$OUT.tmp"

first=1
for bucket in short medium long; do
  CORPUS="$ROOT/corpus/corpus_llama_${bucket}.txt"
  runs_json=""
  for r in 1 2 3 4 5; do
    log=$(mktemp)
    llama-embedding -m "$MODEL" -f "$CORPUS" --pooling mean \
      --embd-output-format array --embd-normalize 2 \
      -ngl 99 > /dev/null 2> "$log" || true
    # Extract llama_perf lines
    prompt_eval_ms=$(grep "prompt eval time" "$log" | head -1 | sed -E 's/.*= +([0-9]+\.[0-9]+) ms \/.*/\1/' || echo "0")
    prompt_tokens=$(grep "prompt eval time" "$log" | head -1 | sed -E 's/.*ms \/ +([0-9]+) tokens.*/\1/' || echo "0")
    total_ms=$(grep "total time" "$log" | head -1 | sed -E 's/.*= +([0-9]+\.[0-9]+) ms \/.*/\1/' || echo "0")
    total_tokens=$(grep "total time" "$log" | head -1 | sed -E 's/.*ms \/ +([0-9]+) tokens.*/\1/' || echo "0")
    n_batches=$(grep "batch_decode: n_tokens" "$log" | wc -l | tr -d ' ')
    max_n_seq=$(grep "batch_decode: n_tokens" "$log" | sed -E 's/.*n_seq = ([0-9]+).*/\1/' | sort -n | tail -1 || echo "0")
    sent_per_s_total=$(python3 -c "print(100.0 / ($total_ms / 1000.0))" 2>/dev/null || echo "0")
    sent_per_s_eval=$(python3 -c "print(100.0 / ($prompt_eval_ms / 1000.0))" 2>/dev/null || echo "0")
    tokens_per_s=$(python3 -c "print($prompt_tokens / ($prompt_eval_ms / 1000.0))" 2>/dev/null || echo "0")
    runs_json+="{\"run\":$r,\"prompt_eval_ms\":$prompt_eval_ms,\"prompt_tokens\":$prompt_tokens,\"total_ms\":$total_ms,\"n_batches\":$n_batches,\"max_n_seq\":$max_n_seq,\"sent_per_s_total\":$sent_per_s_total,\"sent_per_s_eval\":$sent_per_s_eval,\"tokens_per_s\":$tokens_per_s},"
    echo "[llama-v2/$bucket/$r] eval=${prompt_eval_ms}ms (${prompt_tokens}tok) total=${total_ms}ms sent/s_total=$sent_per_s_total" >&2
    rm "$log"
  done
  runs_json="${runs_json%,}"
  if [ "$first" -eq 1 ]; then first=0; else echo "," >> "$OUT.tmp"; fi
  echo "\"$bucket\": { \"runs\": [$runs_json] }" >> "$OUT.tmp"
done
echo "}}" >> "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "wrote $OUT"
