# apple-silicon-embed-bench

Cross-backend embedding inference benchmark for Apple silicon: compares Apple
Neural Engine (ANE), Apple GPU via CoreML, Apple GPU via MLX, and Apple GPU via
llama.cpp Metal — all on the same machine, same model, same corpus.

> **Scope note.** This bench measures one specific deployment: `BAAI/bge-small-en-v1.5`
> (33M params, BERT-12, FP16) converted to CoreML via the standard
> coremltools+`torch.jit.trace` path, traced at fixed batch=1 / seq=512. It does
> **not** test INT8 quantized models, Apple's [`ml-ane-transformers`](https://github.com/apple/ml-ane-transformers) attention rewrite
> (the documented ANE optimum), or larger models (BGE-base / E5-mistral / etc.).
> "ANE is slow" claims below should be read as
> "ANE-via-naive-FP16-coremltools-trace is slow." A different conversion path
> may produce different numbers. We welcome PRs that add those variants.

## TL;DR

Two questions, two tables — because the four backends deploy in different shapes.

### Single-query latency (batch=1)

How fast can each backend answer a one-sentence query? Relevant for interactive search, RAG query embedding.

| Hardware | CoreML ANE | CoreML GPU | CoreML CPU | MLX |
|---|---:|---:|---:|---:|
| M5 Max | 89 | 565 | 58 | 561 |
| M4 Pro | 80 | 173 | 71 | 396 |

Sentences/sec. CoreML rows pad every input to seq=512 (the mlpackage is traced at that fixed shape). MLX b=1 is a 100-sentence loop with one inference call per sentence.

### Throughput (natural batching)

How fast can each backend chew through a corpus? Relevant for indexing, bulk reindex.

| Hardware | llama.cpp Metal (n_seq~66 batched) | MLX (100-in-one-call) |
|---|---:|---:|
| M5 Max | **1,567** | **6,950** |
| M4 Pro | 1,119 | 860 |

Sentences/sec on short bucket (~32 tokens). llama.cpp does internal batching (~66 seq per forward pass). MLX numbers are an extreme batched case (100 in one call); a more realistic b=16 or b=32 would land lower.

Detailed breakdowns by sequence length: [docs/results-m5-max.md](docs/results-m5-max.md) + [community-results/](community-results/).

### Headline findings (with appropriate caveats)

1. **ANE-via-naive-FP16-coremltools-trace is the slowest GPU-class path** on both M4 Pro and M5 Max for this model. Not the fastest. (Different conversion paths, especially INT8 + `ml-ane-transformers`, may differ; not tested here.)
2. **ANE perf is roughly flat across M-generations** (~80-90 sent/s on both M4 Pro and M5 Max). GPU is where Apple is shipping perf gains: 3.3× CoreML GPU and 8× batched MLX from M4 Pro → M5 Max (n=1 cross-gen per chip variant; thermals not controlled).
3. **MLX dominates large-batch throughput** on M5 Max (4× faster than llama.cpp Metal at 100-in-one-call). The gap shrinks on M4 Pro and at more realistic batch sizes.
4. **The MLX-vs-llama gap is partly an Apple-API-exposure state**, not all closeable in llama.cpp code. llama.cpp build 9150 has `has tensor = false` in its Metal init log on macOS 26.5 — Apple's M5 tensor accelerators aren't being used. MLX uses them. A future llama.cpp release that re-enables tensor units may shrink the gap before any kernel work happens.

## Why this exists

A common assumption in the Apple-silicon ML community is that the Neural
Engine (ANE) is the fast efficient inference path. For some workloads
(vision encoders, Whisper) that's true. For transformer text embeddings on
FP16 with the standard HF → coremltools → mlpackage conversion path, **it's not**.

This repo is a reproducible bench so anyone can verify on their own Mac.

## Run the bench

Prereqs:
- macOS 14+ (Sonoma or later; tested on macOS 26.5)
- Apple silicon (M1+)
- Python **3.10 – 3.13** (3.12 is cleanest; 3.13 works via the bench's
  `os._exit(0)` workaround for Apple's coremltools destructor race; 3.14 is
  not supported because torch 2.7.0 has no wheel for it)
- `brew install llama.cpp` (optional but recommended — for the llama.cpp row)

One command:

```bash
git clone https://github.com/petejm/apple-silicon-embed-bench
cd apple-silicon-embed-bench
./bench.sh
```

The script:
1. Creates a Python venv and installs requirements (idempotent — rebuilds venv only if `requirements.txt` changed)
2. Builds the deterministic public-domain corpus
3. Fetches a pre-built CoreML .mlpackage from the GitHub release (one-time, ~60 MB; SHA256-verified)
   - To convert locally instead (only works reliably on M5 Max as of this writing): `REBUILD_MLPACKAGE=1 ./bench.sh`
4. Downloads the bge-small GGUF (one-time, ~67 MB; SHA256-verified)
5. Probes actual device placement for each CoreML compute-unit request (no silent CPU fallback)
6. Runs all backends (CoreML ANE/GPU/CPU, MLX, llama.cpp Metal)
7. Prints a copy-pasteable result block with per-backend status

Total run time: ~5-15 minutes depending on hardware.

## Submit your numbers

Please share results from your hardware:

1. Run `./bench.sh`
2. Copy everything between `===BEGIN RESULT===` and `===END RESULT===` (also saved to `results/RESULT.md`)
3. [Open a new issue](https://github.com/petejm/apple-silicon-embed-bench/issues/new?template=bench-result.md) and paste

Especially valuable: M1 / M1 Pro / M1 Max / M1 Ultra, M2 (all variants),
M3 (all variants), M4 (non-Pro). We have M4 Pro and M5 Max already. The
cross-generation trend is the interesting open question.

## What's in the repo

```
.
├── README.md                  — this file
├── LICENSE                    — Apache 2.0 (covers the code)
├── bench.sh                   — one-command runner with SHA-verified downloads
├── requirements.txt           — Python deps (pinned)
├── bench/
│   ├── convert_bge_coreml.py  — one-time mlpackage conversion (M5-only as of now)
│   ├── bench_coreml.py        — CoreML ANE/GPU/CPU bench (with os._exit workaround)
│   ├── bench_llama_v2.sh      — llama.cpp Metal bench (fail-loud on parse errors)
│   ├── bench_mlx.py           — MLX-embeddings bench (with materialization fix)
│   └── probe_devices.py       — verifies actual ANE/GPU/CPU placement (no silent fallback)
├── corpus/
│   ├── build_corpus.py        — deterministic corpus build
│   ├── sources.txt            — public-domain text (Pride and Prejudice excerpt)
│   └── NOTICE.md              — corpus is PUBLIC DOMAIN, not Apache 2.0
├── docs/
│   └── results-m5-max.md      — canonical M5 Max writeup with caveats
├── community-results/         — crowdsourced data points
└── .github/ISSUE_TEMPLATE/    — bench-result.md template
```

## Methodology

- **Model**: `BAAI/bge-small-en-v1.5` (33M params, BERT-12 encoder, dim=384, FP16). Chosen because it converts cleanly to every backend, is a common production embedder, and is small enough that hardware differences dominate (not model size). HF revision pinned to commit `5c38ec7c`. Conclusions may not transfer to larger / INT8-quantized models.
- **Corpus**: 300 sentences from a fixed public-domain text (Pride and Prejudice), bucketed by length: short (~32 tok), medium (~124 tok), long (~462 tok). 100 sentences per bucket. 19th-century prose; tokenization profile differs from code or modern conversational text.
- **Protocol**: 5 warm runs of 100 sentences per bucket. Drop run 1, take mean of last 4. Cold start timed separately per backend (definitions differ — see results doc).
- **Throughput unit**: sentences/sec, higher is better. For llama.cpp we use the internal `total_time` metric (not wall clock) because process spawn dominates wall for short benches. CoreML pads to seq=512 so effective tok/s differs from sent/s.
- **Device verification**: `compute_units` is a hint, not a guarantee. `probe_devices.py` enumerates actual placement via `MLComputePlan` and writes `results/devices_<unit>.json`; the bench includes this in the result block.
- **Variance**: 4-sample means (after dropping run 1) without explicit ±σ. For a robust headline, prefer the per-run lists in the raw JSON.

Full methodology details: [docs/results-m5-max.md](docs/results-m5-max.md).

## Known issues + gotchas

These bit us during development; documented so they don't bite you:

1. **MLX is lazy**. Calling MLX's tensor-materialize function on a model-output wrapper (BaseModelOutput) does NOT force compute. Materialize the inner tensor (e.g. `.text_embeds`). The bench does this; if you fork it, beware. First MLX bench reported 43K sent/s; real number was 7K once the tensor was forced.
2. **coremltools 9 + Python 3.13 + macOS 26 segfaults** in async destructor cleanup. Workaround: `os._exit(0)` after writing results. Applied in bench scripts. Apple should fix this; Python 3.12 is unaffected.
3. **llama.cpp wall-time != inference time**. Process spawn / Metal pipeline init can take ~500ms. We use llama's `total_time` print instead.
4. **CoreML mlpackage is traced at one fixed shape**. The bundled conversion uses batch=1, seq=512. Re-tracing requires modifying `convert_bge_coreml.py` and may fail on non-M5 silicon. Use `REBUILD_MLPACKAGE=1` only if you want to retrace.
5. **`has tensor = false` in llama.cpp Metal init** on macOS 26.5 + brew build 9150. Apple's M5 tensor accelerators are temporarily disabled in this build. MLX uses them, llama.cpp doesn't — part of the headline MLX-vs-llama gap is this API-state, not closeable code.

## Licensing

- **Code**: Apache 2.0 (see [LICENSE](LICENSE))
- **Corpus** (`corpus/sources.txt`): public domain (see [corpus/NOTICE.md](corpus/NOTICE.md))
- **Documentation**: Apache 2.0

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Most valuable contribution: submit your
bench results from non-M5 hardware via the issue template.
