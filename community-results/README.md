# Community results

This directory will collect curated bench results submitted by community
members on hardware we don't have access to (everything that isn't M5 Max,
currently).

## How it works

1. Run `./bench.sh` from the repo root
2. [Submit your numbers via the issue template](https://github.com/petejm/apple-silicon-embed-bench/issues/new?template=bench-result.md)
3. We periodically aggregate confirmed submissions into the table below and
   into the README's headline result

## Aggregated results

Numbers below are short-bucket throughput (sentences/sec on bge-small at ~32
token average inputs). Higher is better.

| Chip | Memory | macOS | Submitter | CoreML ANE | CoreML GPU | llama.cpp Metal | MLX batched | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| Apple M5 Max | 64 GB | 26.5 | @petejm | 89 | 565 | 1,567 | 6,950 | reference |
| Apple M4 Pro | 24 GB | 26.5 | @petejm | 80 | 173 | 1,119 | 860 | [details](m4-pro-26.5/) |

**Cross-generation insights so far**:
- ANE perf is roughly flat across M-generations (~80-90 sent/s on both M4 Pro and M5 Max).
- M5 Max GPU is much faster than M4 Pro GPU: 3.3× on CoreML, 8× on MLX. Apple is investing in GPU, not ANE.
- llama.cpp Metal scales only ~1.4× M4 Pro → M5 Max because it doesn't yet use M5's new tensor cores. MLX does.

We need more data points — especially M1, M2, M3 across all variants, and macOS 14/15 baselines.

## Submitting raw JSON

Optional: if you want to contribute machine-readable raw data (not just the
markdown block), you can open a PR adding your full `results/` directory
under `community-results/<chip>-<macos>-<short-id>/`. We don't require this —
the issue template is simpler — but it's welcome for rigorous reviewers.
