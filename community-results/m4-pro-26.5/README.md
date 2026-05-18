# Apple M4 Pro — macOS 26.5

Hardware: Mac mini M4 Pro, 24 GB unified memory
macOS: 26.5 (build 25F71)
Python: 3.12.13
Submitter: @petejm
Run date: 2026-05-18

## Result

| Backend | short b=1 | short batched | medium b=1 | medium batched | long b=1 | long batched | Cold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoreML ANE | 79.8 | — | 79.8 | — | 79.8 | — | 1.71 |
| CoreML GPU | 173.2 | — | 172.8 | — | 172.9 | — | 1.63 |
| CoreML CPU | 70.8 | — | 70.9 | — | 70.8 | — | 0.25 |
| MLX-embeddings | 395.5 | **860.4** | 286.5 | **500.2** | 118.1 | 115.7 | 0.39 |
| llama.cpp Metal | — | **1,119.4** | — | **419.9** | — | 132.9 | — |

## Notes

- Pre-built `.mlpackage` was rsynced from M5 Max (canonical bench machine) because fresh CoreML conversion via `coremltools 9.0` + `torch 2.7.0` failed on this machine with `TypeError: only 0-dimensional arrays can be converted to Python scalars` in coremltools' torch frontend. The .mlpackage is functionally identical; only the conversion step was skipped.
- MLX-embeddings + mlx-vlm + mlx-lm + mlx-audio were installed with `--no-deps` to avoid the `mlx-vlm 0.4.4 requires transformers>=5.1` constraint chain that fights with our `transformers==4.57.6` pin (the convert script needs 4.x).
- Runtime deps (Pillow, fastapi, opencv-python, etc.) were installed manually after the `--no-deps` install. The next `bench.sh` revision should automate this.

## Cross-generation comparison vs M5 Max

| Backend / bucket | M5 Max | M4 Pro | M5 / M4 |
|---|---:|---:|---:|
| CoreML ANE (b=1 fixed) | 89.0 | 79.8 | 1.1× |
| CoreML GPU (b=1 fixed) | 564.5 | 173.2 | **3.3×** |
| llama.cpp Metal short batched | 1567 | 1119 | 1.4× |
| llama.cpp Metal medium batched | 597 | 420 | 1.4× |
| llama.cpp Metal long batched | 197 | 133 | 1.5× |
| MLX short batched | **6,950** | **860** | **8.1×** |
| MLX medium batched | 1,747 | 500 | 3.5× |

Key cross-generation findings:
1. **ANE perf is roughly flat across generations.** M5 Max is ~1.1× M4 Pro at the ANE workload. This is consistent with Apple not having significantly upgraded the ANE for transformer workloads in the M5 generation.
2. **GPU is where Apple invested in M5.** CoreML GPU is 3.3× faster, MLX batched is 8× faster. The new tensor cores in M5 (which llama.cpp had to disable in early macOS 26 due to API regressions) are likely the reason.
3. **llama.cpp's gains are modest (~1.4×)** because it can't use the new M5 tensor cores yet (`has tensor = false` in init log on macOS 26.5 + build 9150). MLX is using them; that's where the 8× came from.

This reinforces Finding 3 from the M5 docs: closing llama.cpp's gap to MLX is the actual upstream opportunity. The gap widens as Apple ships new GPU silicon.
