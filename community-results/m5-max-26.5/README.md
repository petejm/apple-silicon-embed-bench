# Apple M5 Max — macOS 26.5 (reference machine, 10-run variance sweep)

Hardware: MacBook Pro M5 Max, 64 GB unified memory
macOS: 26.5 (build 25F71)
Python: 3.12.13
llama.cpp: brew build 9150, ggml 0.11.1, `has tensor = false` in Metal init
Submitter: @petejm
Run date: 2026-05-18

**Canonical writeup with full findings + methodology**: [../../docs/results-m5-max.md](../../docs/results-m5-max.md). This directory holds raw JSON + summary for symmetry with other community-results.

## 10-run variance sweep — sentences/sec, mean [min…max] σ=N (CV%)

| backend | short b=1 | medium b=1 | long b=1 | short batched | medium batched | long batched |
|---|---|---|---|---|---|---|
| CoreML ANE | 89 [89…90] σ=0 (0.2%) | 89 [86…89] σ=1 (1.0%) | 89 [87…89] σ=1 (0.9%) | — | — | — |
| CoreML GPU | 572 [535…578] σ=13 (2.3%) | 562 [446…581] σ=42 (**7.4%**) | 564 [443…579] σ=43 (**7.6%**) | — | — | — |
| CoreML CPU | 59 [59…59] σ=0 (0.5%) | 59 [59…60] σ=0 (0.6%) | 59 [59…60] σ=0 (0.5%) | — | — | — |
| MLX (b=1) | 562 [544…567] σ=7 (1.2%) | 528 [512…533] σ=6 (1.1%) | 329 [313…332] σ=6 (1.7%) | — | — | — |
| MLX (b=100) | — | — | — | **3,099 [3070…3125] σ=18 (0.6%)** | 1,647 [1634…1654] σ=6 (0.3%) | 325 [319…326] σ=2 (0.6%) |
| llama.cpp Metal | — | — | — | 1,176 [1146…1187] σ=12 (1.0%) | 500 [467…524] σ=20 (4.0%) | 199 [196…203] σ=2 (1.2%) |

## Raw JSON

The eight JSON files in this directory are the first run (`runs/001/`) of the 10-run sweep. Per-run data for all 10 runs is preserved on the canonical machine; submit a PR if you want the full set added here.

- `coreml_ane.json` / `coreml_gpu.json` / `coreml_cpu.json` — CoreML inference results (5 internal runs per bucket × 3 buckets)
- `mlx_embeddings.json` — MLX-embeddings b=1 + b=100 results
- `llama_metal.json` — llama.cpp Metal results (per-run status field; all `ok` here)
- `devices_ane.json` / `devices_gpu.json` / `devices_cpu.json` — MLComputePlan device-placement verification (291/315 ops on ANE for the ane row; methodology evidence that the reported ANE number is real ANE, not silent CPU fallback)

## Observations specific to this hardware

- **Most variable backend**: CoreML GPU at medium/long buckets (CV 7.4-7.6%, range 443-581). Likely OS-level GPU contention (window compositor, IDE, browser) — the same workload on the headless Mac mini M4 Pro produced CV < 0.5%. If you're benchmarking on a MacBook, expect noisier CoreML GPU numbers.
- **Most stable**: CoreML ANE short b=1 at CV 0.2% (range 89.0-89.6). Apple's runtime appears to give very deterministic ANE timing.
- **MLX batched-short retraction**: an earlier single-run snapshot showed 6,950 sent/s. 10 runs produced 3,099 ± 18. The 6,950 was a measurement artifact from an earlier code path; it's been retracted everywhere in this repo.
- **Real M5/M4 cross-gen ratios** (with the now-corrected M5 numbers and the M4 Pro 10-run sweep):
  - ANE: 1.11× (essentially flat)
  - CoreML GPU: 3.31× (M5 invested here)
  - MLX batched-short: 3.57× (uses M5 tensor cores)
  - llama.cpp Metal: 1.26-1.76× (no tensor cores yet)
- **Device verification**: ANE 291/315 ops on Neural Engine, 24 on CPU (92% ANE). GPU 315/315 on Metal GPU. CPU 315/315 on CPU. The "ANE is slow" finding is real-ANE-slow, not silent-fallback-slow.

## See also

- [../../docs/results-m5-max.md](../../docs/results-m5-max.md) — full methodology + findings + caveats
- [../m4-pro-26.5/](../m4-pro-26.5/) — Mac mini M4 Pro 10-run sweep for cross-gen comparison
