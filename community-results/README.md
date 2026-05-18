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

*(empty so far — be the first!)*

| Chip | Memory | macOS | Submitter | CoreML ANE | CoreML GPU | llama.cpp Metal | MLX batched | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| Apple M5 Max | 64 GB | 26.5 | @petejm | 89 | 565 | 1,567 | 6,950 | reference |

Numbers above are short-bucket throughput (sentences/sec on bge-small at ~32
token average inputs). Higher is better.

## Submitting raw JSON

Optional: if you want to contribute machine-readable raw data (not just the
markdown block), you can open a PR adding your full `results/` directory
under `community-results/<chip>-<macos>-<short-id>/`. We don't require this —
the issue template is simpler — but it's welcome for rigorous reviewers.
