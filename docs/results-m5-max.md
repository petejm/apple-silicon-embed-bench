# CoreML / ANE / MLX / llama.cpp Embedding Benchmark — M5 Max

*Canonical results for the reference machine. Reproduce on your own Mac via `./bench.sh` at the repo root.*

## Scope

This bench measures **one specific deployment shape**: `BAAI/bge-small-en-v1.5` (33M, BERT-12, FP16) converted to CoreML via standard `coremltools.convert(torch.jit.trace(...))` at fixed batch=1 / seq=512. It does **not** test:
- INT8 quantized models (ANE's documented sweet spot)
- Apple's [`ml-ane-transformers`](https://github.com/apple/ml-ane-transformers) attention rewrite for ANE
- Larger models (BGE-base, BGE-large, E5-mistral, etc.)
- Realistic-batch shapes for MLX (b=16, b=32) — only b=1 and b=100
- INT8 GGUF quantizations in llama.cpp
- Sustained-throughput conditions (thermal-controlled long runs)
- Roofline analysis (theoretical FP16 / memory-bandwidth ceiling)

Conclusions apply to **this stack**, not to ANE / GPU / Metal in general. PR welcome to add the missing variants.

## TL;DR

**ANE-via-naive-FP16-coremltools-trace is the slowest GPU-class path on M5 Max for this model.** Not the fastest. The widely-held assumption that ANE is the fast efficient path for inference on Apple silicon doesn't generalize from vision/ASR (where it's true) to text-embedding transformers under this conversion pipeline.

- CoreML ANE: ~89 sent/s
- CoreML CPU: ~59 sent/s
- CoreML GPU: 464-565 sent/s
- llama.cpp Metal: 197-1567 sent/s (depends on seq len, batched internally)
- **MLX-embeddings: 282-6950 sent/s** (dominates at short/medium)

llama.cpp's BERT-embed Metal path is leaving 4-7× perf on the table vs MLX on identical hardware running the same model. That gap is the tractable upstream optimization opportunity that fell out of this work.

## Setup

- **Hardware**: Apple M5 Max, 18-core (6P+12E), 64GB unified memory
- **OS**: macOS 26.5 (build 25F71)
- **llama.cpp**: brew build 9150, ggml 0.11.1, Metal backend hot
- **Python**: 3.13.13 (homebrew)
- **coremltools**: 9.x
- **mlx-embeddings**: latest PyPI at bench time
- **Model**: `BAAI/bge-small-en-v1.5` (33M params, BERT encoder, dim=384)
  - GGUF: `bge-small-en-v1.5-f16.gguf` (CompendiumLabs)
  - mlpackage: traced at batch=1, seq=512 fixed (ANE-friendly form)
- **Corpus**: 300 sentences from Pride and Prejudice (public domain), bucketed:
  - short (~32 tokens avg)
  - medium (~124 tokens avg)
  - long (~462 tokens avg)
  - 100 sentences per bucket
- **Protocol**: 5 runs of 100 sentences, drop run 1, mean of last 4

## Reproducibility caveat (post-publication)

A fresh-clone re-run of the hardened bench on the same M5 Max produced
**MLX batched-short = 3,099 sent/s**, not the 6,950 reported in the table
below (b=1 numbers reproduced exactly: 566 vs 561 originally). Same versions,
same model, same corpus. The peak number is thermally sensitive — chassis
state, sustained-load history, and background processes shift it. Treat the
6,950 figure as an **upper bound observed**; **3,000-7,000 is the realistic
range** on this hardware for that workload.

The *direction* of every finding (ANE slowest GPU-class path, MLX dominates
batched short/medium, llama.cpp's Metal BERT-embed leaves real perf on the
table) is robust across re-runs. Specific magnitude factors at the extreme
end (the 8× cross-gen MLX claim, the 78× ANE-vs-MLX claim) should be read
as point estimates, not population means. Variance / sustained-throughput
measurements are an outstanding methodology gap to close (see Open issues).

## Results

### Sentences/second (higher is better)

| Backend | short b=1 | short batched | medium b=1 | medium batched | long b=1 | long batched | Cold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoreML ANE (seq=512 fixed) | 89.0 | — | 89.0 | — | 89.0 | — | 1.45 |
| CoreML GPU (seq=512 fixed) | 564.5 | — | 546.5 | — | 464.3 | — | 1.57 |
| CoreML CPU (seq=512 fixed) | 58.2 | — | 58.6 | — | 58.5 | — | 0.22 |
| MLX-embeddings | 561.1 | **6,949.6** | 523.2 | **1,747.2** | 287.7 | 282.0 | 1.19 |
| llama.cpp Metal (n_seq~66 internally) | — | **1,567** | — | **597** | — | 197 | — |

Notes:
- ANE row uses fixed seq=512 padding (the only ANE-compatible form for fixed-shape transformers).
- CoreML mlpackage was traced at batch=1; per-sentence loop reflects realistic single-query inference, but ALSO blocks higher batched throughput. Re-tracing per batch size is a deployment burden.
- llama.cpp does its own batching internally — typical n_seq=66 in this corpus. Throughput from llama's `total_time` metric, not wall clock (wall is dominated by process spawn).
- MLX-embeddings: `b=1` runs `generate()` in a 100-sentence loop; `batched` sends all 100 in one call.

### Device verification

- **ANE**: 291 ops on Neural Engine, 24 on CPU = 92% ANE. Real, not a silent fallback.
- **GPU**: All 315 ops on `MLGPUComputeDevice`.
- **CPU**: Confirmed CPU-only path.
- ANE/GPU/CPU placement captured via `MLComputePlan.preferredDeviceForOp` (in-process, isolated probe script). The reported ANE throughput is real ANE perf, not silent CPU fallback.

### Crossover for CoreML GPU vs llama.cpp Metal

CoreML GPU is faster ONLY at long sequences (~462 tokens). At short/medium it's slower than llama.cpp's batched path. Crossover point is around 185-200 tokens.

**Why**: CoreML processes the full padded seq=512 every call (290K tok/s effective). llama.cpp processes only real tokens (50-200K tok/s, varies with seq). Short docs: llama wins by skipping padding compute. Long docs: CoreML's static graph compilation pays off.

This is the only legitimate niche for a CoreML embedding path — long-doc embedding workloads — and the gap is 2.4× at the high end. Not enough to justify the toolchain cost vs MLX or vs improving llama.cpp's Metal kernels directly.

## Findings

### Finding 1: ANE is not the embedding accelerator we hoped

At every measured seq length on M5 Max, ANE runs bge-small at ~89 sentences/sec. CoreML GPU on the same model is 5-6× faster. llama.cpp Metal at short batched is 17× faster. MLX-embeddings batched is 78× faster. **ANE is the slowest GPU-class device on this chip for transformer embedding inference.**

This is the opposite of the widely-held belief that ANE is the "fast efficient path" for inference on Apple silicon. ANE's strengths are vision models (CNNs, ViTs at int8) and small fixed-shape encoders like Whisper. For text-embedding transformers in FP16, it underperforms the GPU.

### Finding 2: MLX is the actual fast path on Apple silicon

MLX-embeddings hits ~7K sent/s on short batched workloads — 4.4× faster than llama.cpp Metal and 12× faster than CoreML GPU on the same model. MLX is Apple's first-party ML framework, actively developed, uses Metal kernels with aggressive fusion. It bypasses the CoreML graph-compile layer (no .mlpackage required) and gets straight to optimal Metal shaders.

If you want to ship the fastest embedding path on Apple silicon today, you wrap MLX or you out-engineer MLX's Metal kernels. CoreML is a detour.

### Finding 3: llama.cpp's Metal embedding path is leaving 4-7× on the table — *partly closeable, partly Apple's API state*

On the same hardware running the same model, MLX is 4-7× faster than llama.cpp's Metal embedding path at extreme batched workloads (100-in-one-call).

**Important caveat — the gap is partly Apple's API exposure, not fully closeable in llama.cpp code.** The Metal init log on macOS 26.5 + brew build 9150 reports `has tensor = false`. Apple's M5 tensor accelerators are temporarily disabled in this build of llama.cpp; MLX uses them. A future llama.cpp release that re-enables tensor units (whisper.cpp b8920 era) may shrink this gap substantially before any kernel work happens.

Even with tensor units accounted for, plausible code-side wins exist:
- Better attention layout for BERT-style bidirectional models (no KV cache, no causal mask — the decoder-oriented path may be doing wasted work)
- Larger ubatch sizes (current default 512 may be too small for embedding batched workloads where total input tokens cap at <50K)
- Per-op profiling to find CPU-side bottlenecks (pooling, normalize, output gather)

This is a more interesting upstream contribution than a new CoreML backend. Doesn't require a new dependency, doesn't fork the build matrix, benefits 100% of Apple users not just those who run convert scripts. The investigation plan: see the [llama.cpp BERT-embed Metal Perf PRP](https://github.com/petejm/apple-silicon-embed-bench/blob/main/docs/) (TODO — link to PRP once it lives somewhere public).

### Finding 4: coremltools + Python 3.13 + macOS 26 is a minefield

During this work, Python crashed **5 times in 12 minutes** with `EXC_BAD_ACCESS` in `MLE5ExecutionStream.resetQueue` calling `_PyObject_Free` without holding the GIL (libdispatch worker thread). Apple's CoreML async cleanup queue calls back into Python's allocator without acquiring the GIL. Known bug in the wild for this stack combination.

Worked around by patching the bench script to call `os._exit(0)` after writing results, bypassing the destructor path that triggers the segfault. Works, but the bug remains; any production CoreML+Python pipeline on this stack would need the same hack or a downgrade to Python 3.12.

Don't ship CoreML-Python anywhere durable until Apple fixes this.

### Finding 5: CoreML mlpackage is operationally awkward for variable-length workloads

The .mlpackage is traced at one fixed (batch, seq) shape. To support both b=1 (latency-sensitive query) and b=64 (throughput-oriented reindex), you need TWO models on disk. Same for short vs long seq. The deployment matrix multiplies fast. llama.cpp's runtime tensor shapes are dynamic by design — no per-batch-shape compilation needed. MLX is similarly dynamic.

This alone would kill any CoreML-as-llama.cpp-backend story even if the perf had been favorable.

## Reproducibility

All bench scripts and corpus are in the repo. Reproduce on your own hardware:

```bash
./bench.sh
```

Result files land in `results/`:
- `coreml_ane.json` / `coreml_gpu.json` / `coreml_cpu.json` — CoreML data
- `llama_metal.json` — llama.cpp data
- `mlx_embeddings.json` — MLX data

The script prints a copy-pasteable result block at the end. [Submit your numbers](https://github.com/petejm/apple-silicon-embed-bench/issues/new?template=bench-result.md) so we can build a cross-generation comparison.

## Gotchas (durable lessons)

1. **MLX is lazy**. Calling MLX's tensor-materialize function on a model-output wrapper (BaseModelOutput) does NOT force compute. You must materialize the inner tensor — e.g., the `text_embeds` field — explicitly. Silently returns fake-fast results otherwise. First MLX bench reported 43K sent/s; real number was 7K once the actual tensor was forced.

2. **coremltools 9 + Python 3.13 + macOS 26 segfaults** in async destructor cleanup. Workaround: `os._exit(0)` after writing results.

3. **llama.cpp wall-time != inference time**. Process spawn (Metal pipeline init) is ~500ms — dominates wall clock for short benches. Use `llama_perf_context_print` totals instead.

4. **CoreML `compute_units` is a hint, not a guarantee.** Always verify with `MLComputePlan.preferredDeviceForOp` or equivalent. We did; ANE actually ran 92% of ops on ANE, so the 89 sent/s number is real ANE perf, not silent CPU fallback.
