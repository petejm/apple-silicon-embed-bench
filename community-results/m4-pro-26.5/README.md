# Apple M4 Pro — macOS 26.5 (10-run variance sweep)

Hardware: Mac mini M4 Pro, 24 GB unified memory
macOS: 26.5 (build 25F71)
Python: 3.12.13
Submitter: @petejm
Run date: 2026-05-18

## 10-run variance sweep — sentences/sec, mean [min…max] σ=N (CV%)

| backend | short b=1 | medium b=1 | long b=1 | short batched | medium batched | long batched |
|---|---|---|---|---|---|---|
| CoreML ANE | 80 [80…80] σ=0 (**0.1%**) | 80 [80…80] σ=0 (0.2%) | 80 [80…80] σ=0 (0.2%) | — | — | — |
| CoreML GPU | 173 [173…175] σ=1 (0.4%) | 173 [172…175] σ=1 (0.5%) | 173 [172…175] σ=1 (0.4%) | — | — | — |
| CoreML CPU | 73 [71…74] σ=1 (1.6%) | 73 [71…75] σ=1 (1.4%) | 73 [71…75] σ=1 (1.6%) | — | — | — |
| MLX (b=1) | 384 [381…387] σ=2 (0.6%) | 277 [275…279] σ=1 (0.5%) | 115 [115…116] σ=0 (0.4%) | — | — | — |
| MLX (b=100) | — | — | — | 869 [868…872] σ=1 (**0.1%**) | 504 [503…505] σ=1 (0.1%) | 117 [117…118] σ=0 (0.2%) |
| llama.cpp Metal | — | — | — | 933 [927…958] σ=9 (1.0%) | 351 [348…365] σ=5 (1.5%) | 113 [112…115] σ=1 (0.8%) |

## Notes

- **Variance is dramatically tighter than M5 Max MacBook**. Most CVs under 1%, MLX is essentially noise-free (σ=1 sent/s on a 869 sent/s mean). This is consistent with the Mac mini's chassis cooling + lack of background-app contention vs a laptop running a browser / IDE / window manager.
- Pre-built `.mlpackage` was downloaded from the GitHub release v0.1.0 (SHA256-verified). Conversion was not attempted on this hardware because `coremltools 9 + torch 2.7` produces a `TypeError: only 0-dimensional arrays can be converted to Python scalars` on M4 Pro for reasons unrelated to the bench. The .mlpackage was traced once on M5 Max and is the canonical artifact for cross-machine comparison.
- All deps pinned per the repo's `requirements.txt`. mlx-embeddings + mlx-vlm installed with `--no-deps` per `bench.sh`'s automation, then transitive runtime deps (Pillow, fastapi, opencv-python, miniaudio, llguidance, uvicorn, datasets) pulled separately.

## Cross-generation comparison (10-run means)

| Backend / bucket | M5 Max | M4 Pro | M5 / M4 ratio |
|---|---:|---:|---:|
| CoreML ANE (b=1, short) | 89 | 80 | 1.11× |
| CoreML GPU (b=1, short) | 572 | 173 | **3.31×** |
| CoreML CPU (b=1, short) | 59 | 73 | 0.81× (M4 wins!) |
| MLX (b=1, short) | 562 | 384 | 1.46× |
| MLX (b=100, short) | 3,099 | 869 | **3.57×** |
| llama.cpp Metal (short batched) | 1,176 | 933 | 1.26× |
| llama.cpp Metal (medium batched) | 500 | 351 | 1.42× |
| llama.cpp Metal (long batched) | 199 | 113 | 1.76× |

**Key cross-generation findings**:

1. **ANE perf is nearly flat across generations** (1.11× M5/M4). Apple is not investing in ANE for transformer text-embedding workloads. This is the headline finding for "is ANE worth optimizing for?": answer, not on this trajectory.
2. **GPU is where Apple is shipping perf**. CoreML GPU 3.3× and MLX batched 3.6× M4 → M5. Likely the new tensor cores in M5 + faster memory.
3. **llama.cpp Metal scales modestly** (1.26-1.76× depending on workload). llama.cpp can't yet use M5's tensor cores (`has tensor = false` in init log on macOS 26.5 + brew build 9150), so it's missing the headline GPU gains.
4. **CoreML CPU is faster on M4 Pro than M5 Max** (73 vs 59 sent/s). Surprising. Possibly: M5 Max's heterogeneous CPU (6P+12E) allocates fewer P-cores to a single-threaded CoreML workload when other compute units are queued. Not investigated further.

## Earlier (single-run) notes

A previous version of this writeup had a single-run snapshot. The 10-run sweep above supersedes it. Direction of findings is unchanged but the specific magnitudes are tighter and more defensible.
