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

Numbers below are 10-run means of short-bucket throughput (sentences/sec on
bge-small at ~32 token average inputs). Higher is better. Single-run
submissions are noted as such.

| Chip | Memory | macOS | Submitter | CoreML ANE | CoreML GPU | llama.cpp Metal | MLX batched | n | Details |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| Apple M5 Max | 64 GB | 26.5 | @petejm | 89 | 572 | 1,176 | 3,099 | 10 | [m5-max-26.5/](m5-max-26.5/) |
| Apple M4 Pro | 24 GB | 26.5 | @petejm | 80 | 173 | 933 | 869 | 10 | [m4-pro-26.5/](m4-pro-26.5/) |

**Cross-generation insights so far**:
- ANE perf is roughly flat across M-generations (1.11× M4 Pro → M5 Max).
- CoreML GPU scales 3.3× and batched MLX scales 3.6× M4 Pro → M5 Max. Apple is investing in GPU, not ANE.
- llama.cpp Metal scales 1.26-1.76× M4 Pro → M5 Max because it doesn't yet use M5's new tensor cores (`has tensor = false` in init log on macOS 26.5 + brew build 9150). MLX does.
- Mac mini is dramatically more thermally stable than MacBook (variance CVs typically <1% on mini vs up to 7.6% on MacBook for the same workload).

> **Correction note**: this table previously showed M5 Max MLX batched at
> 6,950 sent/s, which a 10-run variance sweep could not reproduce. The 6,950
> was a measurement artifact from an earlier version of `bench_mlx.py` that
> didn't fully materialize MLX's lazy graph; that bug was fixed but the
> headline numbers weren't re-measured at the time. The corrected number is
> 3,099 ± 18 (CV 0.6%).

We need more data points — especially M1, M1 Pro/Max/Ultra, M2 (all variants), M3 (all variants), M4 (base + Max), and macOS 14/15 baselines.

## Submitting raw JSON

Optional: if you want to contribute machine-readable raw data (not just the
markdown block), you can open a PR adding your full `results/` directory
under `community-results/<chip>-<macos>-<short-id>/`. We don't require this —
the issue template is simpler — but it's welcome for rigorous reviewers.
