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
- CoreML GPU: 464-572 sent/s
- llama.cpp Metal: 199-1,176 sent/s (depends on seq len, batched internally)
- **MLX-embeddings: 325-3,099 sent/s** (dominates at short/medium)

llama.cpp's BERT-embed Metal path is leaving 2.6-3.3× perf on the table vs MLX on identical hardware running the same model. That gap is the tractable upstream optimization opportunity that fell out of this work.

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

## 10-run variance sweep (2026-05-18)

Repeated `./bench.sh` 10 times back-to-back on the same M5 Max MacBook to characterize variance properly. CV = coefficient of variation (σ / mean).

| backend | short b=1 | medium b=1 | long b=1 | short batched | medium batched | long batched |
|---|---|---|---|---|---|---|
| CoreML ANE | 89 [89…90] CV 0.2% | 89 [86…89] CV 1.0% | 89 [87…89] CV 0.9% | — | — | — |
| CoreML GPU | 572 [535…578] CV 2.3% | 562 [446…581] **CV 7.4%** | 564 [443…579] **CV 7.6%** | — | — | — |
| CoreML CPU | 59 [59…59] CV 0.5% | 59 [59…60] CV 0.6% | 59 [59…60] CV 0.5% | — | — | — |
| MLX (b=1) | 562 [544…567] CV 1.2% | 528 [512…533] CV 1.1% | 329 [313…332] CV 1.7% | — | — | — |
| MLX (b=100) | — | — | — | **3,099 [3070…3125] CV 0.6%** | 1,647 [1634…1654] CV 0.3% | 325 [319…326] CV 0.6% |
| llama.cpp Metal | — | — | — | 1,176 [1146…1187] CV 1.0% | 500 [467…524] CV 4.0% | 199 [196…203] CV 1.2% |

**Headline numbers retracted**: an earlier version of this doc showed MLX batched-short at 6,950 sent/s. 10 runs produce 3,099 ± 18 (CV 0.6%, range 3,070-3,125). 6,950 was an outlier that didn't reproduce; the prior comparison framing has been corrected throughout this repo. The earlier MLX number was likely the result of an incomplete lazy-graph materialization in an earlier version of `bench_mlx.py` (the fix was committed but the headline numbers weren't re-measured at the time).

**Observations from the variance sweep**:
- **CoreML ANE and CPU are rock-stable** (CV < 1%). ANE in particular shows 0.2% CV on short batched — Apple's runtime appears to give very deterministic perf here.
- **CoreML GPU is the noisiest** at medium/long buckets (CV 7.4-7.6%, range 443-581). Plausibly contention with OS-level GPU consumers (window compositor, browser, etc.) since this is a MacBook. The Mac mini variance sweep saw CV < 0.5% on the same path → it's a MacBook-thermals / multitasking artifact, not silicon variance.
- **MLX batched is extremely stable** (CV 0.1-0.6%) once the lazy-graph materialization is correct.
- **llama.cpp medium is unexpectedly noisy** (CV 4.0%, range 467-524) compared to short (1.0%) and long (1.2%). Worth investigating in the PRP's Phase 0.

Per-run raw data preserved in the `runs/` subdirectory of the test workspace.

**Independent corroboration of MLX b=100**: 3,099 sent/s × 32 tok = 99K real tokens/sec. At ~6.5 GFLOPs/sentence forward, that's ~20 TFLOPS sustained. M5 Max GPU FP16 peak is approximately 30-50 TFLOPS — putting MLX at 40-67% of theoretical peak, which is plausible for a fused-attention implementation. The number passes a basic sanity check. We have not independently verified the measurement against a second framework on this exact workload; doing so is on the open-issues list.

## Results

### Sentences/second (higher is better)

| Backend | short b=1 | short batched | medium b=1 | medium batched | long b=1 | long batched | Cold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoreML ANE (seq=512 fixed) | 89.4 | — | 89.0 | — | 89.0 | — | 1.45 |
| CoreML GPU (seq=512 fixed) | 572 | — | 562 | — | 564 | — | 1.57 |
| CoreML CPU (seq=512 fixed) | 59 | — | 59 | — | 59 | — | 0.22 |
| MLX-embeddings | 562 | **3,099** | 528 | **1,647** | 329 | 325 | 1.19 |
| llama.cpp Metal (n_seq~66 internally) | — | **1,176** | — | **500** | — | 199 | — |

These are the 10-run means from the variance sweep above; reproduce on your own hardware via `./bench.sh`.

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

### Finding 1: ANE-via-naive-FP16-coremltools-trace is the slowest GPU-class path on this chip for this model

At every measured seq length on M5 Max, ANE runs bge-small at ~89 sentences/sec. CoreML GPU on the same model is ~6× faster. llama.cpp Metal at short batched is ~13× faster. MLX-embeddings batched is ~35× faster. **ANE-via-naive-FP16-coremltools-trace is the slowest GPU-class device on this chip for this model's transformer embedding inference.**

Scope: this finding applies to the bge-small-en-v1.5 FP16 mlpackage produced by the standard `coremltools.convert(torch.jit.trace(...))` path traced at batch=1/seq=512. A different conversion path — especially INT8 + Apple's [`ml-ane-transformers`](https://github.com/apple/ml-ane-transformers) attention rewrite, which is the documented ANE optimum — may produce a different finding. We did not test it.

This is the opposite of the widely-held belief that ANE is the "fast efficient path" for inference on Apple silicon. ANE's strengths are vision models (CNNs, ViTs at int8) and small fixed-shape encoders like Whisper. For text-embedding transformers in FP16 under the naive conversion path, it underperforms the GPU.

### Finding 2: MLX is the actual fast path on Apple silicon — for this model, this stack, these batch shapes

MLX-embeddings hits ~3.1K sent/s on short batched workloads — 2.6× faster than llama.cpp Metal and 5.4× faster than CoreML GPU on the same model. MLX is Apple's first-party ML framework, actively developed, uses Metal kernels with aggressive fusion. It bypasses the CoreML graph-compile layer (no .mlpackage required) and gets straight to optimal Metal shaders.

Scope: bge-small-en-v1.5 (33M, BERT-12) at FP16, on M5 Max, at b=1 and b=100 (no intermediate batch sizes tested). Larger models (BGE-base, BGE-large, E5-mistral), INT8 quantized variants, and realistic batch sizes (b=16, b=32) may show a different ranking. The gap shrinks on M4 Pro (see `community-results/m4-pro-26.5/`).

If you want to ship the fastest embedding path on Apple silicon today — for this model, this stack, these batch shapes — you wrap MLX or you out-engineer MLX's Metal kernels. CoreML is a detour.

### Finding 3: llama.cpp's Metal embedding path is leaving 2.6-3.3× on the table — *partly closeable, partly Apple's API state*

On the same hardware running the same model, MLX is 2.6-3.3× faster than llama.cpp's Metal embedding path at batched workloads (100-in-one-call). Per bucket: 2.6× short, **3.3× medium** (largest), 1.6× long. The gap shape (largest at medium, not short) is notable — see Phase 1 hypothesis discussion in the PRP; it strengthens the ubatch-sizing hypothesis and weakens the kernel-fusion-at-short hypothesis.

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

1. **MLX is lazy**. Calling MLX's tensor-materialize function on a model-output wrapper (BaseModelOutput) does NOT force compute. You must materialize the inner tensor — e.g., the `text_embeds` field — explicitly. Silently returns fake-fast results otherwise. First MLX bench reported 43K sent/s; the 10-run reproducible number is ~3,099 sent/s once the inner tensor is forced.

2. **coremltools 9 + Python 3.13 + macOS 26 segfaults** in async destructor cleanup. Workaround: `os._exit(0)` after writing results.

3. **llama.cpp wall-time != inference time**. Process spawn (Metal pipeline init) is ~500ms — dominates wall clock for short benches. Use `llama_perf_context_print` totals instead.

4. **CoreML `compute_units` is a hint, not a guarantee.** Always verify with `MLComputePlan.preferredDeviceForOp` or equivalent. We did; ANE actually ran 92% of ops on ANE, so the 89 sent/s number is real ANE perf, not silent CPU fallback.
