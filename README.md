# apple-silicon-embed-bench

Cross-backend embedding inference benchmark for Apple silicon: compares Apple
Neural Engine (ANE), Apple GPU via CoreML, Apple GPU via MLX, and Apple GPU via
llama.cpp Metal — all on the same machine, same model, same corpus.

## TL;DR — ANE is the slow path

Bench results from `BAAI/bge-small-en-v1.5` (BERT-12, 33M params, FP16):

| Hardware | Backend | short (~32 tok) | medium (~124 tok) | long (~462 tok) |
|---|---|---:|---:|---:|
| M5 Max | CoreML ANE | 89 | 89 | 89 |
| M5 Max | CoreML GPU | 565 | 547 | 464 |
| M5 Max | llama.cpp Metal | **1,567** | 597 | 197 |
| M5 Max | **MLX-embeddings (batched)** | **6,950** | **1,747** | 282 |
| M4 Pro | CoreML ANE | 80 | 80 | 80 |
| M4 Pro | CoreML GPU | 173 | 173 | 173 |
| M4 Pro | llama.cpp Metal | **1,119** | 420 | 133 |
| M4 Pro | MLX-embeddings (batched) | 860 | 500 | 116 |

Sentences/second, higher is better. Full tables at [docs/results-m5-max.md](docs/results-m5-max.md) + [community-results/](community-results/).

**Headline findings**:
1. ANE is the *slowest* GPU-class path on both M4 Pro and M5 Max for transformer embedding inference. Not the fastest.
2. **ANE perf is roughly flat across M-generations** (~80-90 sent/s on both M4 Pro and M5 Max). GPU is where Apple is shipping perf gains: 3.3× CoreML GPU and 8× batched MLX from M4 Pro → M5 Max.
3. MLX dominates batched embedding on M5 Max (4× faster than llama.cpp Metal). On M4 Pro the gap closes because llama.cpp doesn't depend on M5's new tensor cores (which MLX uses).
4. llama.cpp's BERT-embed Metal kernels are leaving 4-7× perf on the table on the latest hardware — a tractable upstream optimization opportunity.

Detailed analysis: [docs/results-m5-max.md](docs/results-m5-max.md).

## Why this exists

A common assumption in the Apple-silicon ML community is that the Neural
Engine (ANE) is the fast efficient inference path. For some workloads
(vision encoders, Whisper) that's true. For transformer text embeddings on
FP16, **it's not** — and the gap to the GPU is large.

This repo is a reproducible bench so anyone can verify on their own Mac.

## Run the bench

Prereqs:
- macOS 14+ (Sonoma or later; tested on macOS 26.5)
- Apple silicon (M1+)
- Python **3.11 or 3.12** (NOT 3.13 — coremltools 9 + Python 3.13 has a CoreML
  destructor race that crashes mid-run; bench works around it via `os._exit(0)`
  but Apple should fix this upstream)
- `brew install llama.cpp` (optional but recommended — for the llama.cpp row)

One command:

```bash
git clone https://github.com/petejm/apple-silicon-embed-bench
cd apple-silicon-embed-bench
./bench.sh
```

The script:
1. Creates a Python venv and installs requirements
2. Builds the deterministic public-domain corpus
3. Converts bge-small to a CoreML .mlpackage (one-time, ~30s)
4. Downloads the bge-small GGUF for llama.cpp (one-time, ~67MB)
5. Runs all four backends (CoreML ANE/GPU/CPU, MLX, llama.cpp Metal)
6. Prints a copy-pasteable result block at the end

Total run time: ~5-10 minutes on M5 Max, more on older silicon.

## Submit your numbers

Please share results from your hardware:

1. Run `./bench.sh`
2. Copy everything between `===BEGIN RESULT===` and `===END RESULT===`
3. [Open a new issue](https://github.com/petejm/apple-silicon-embed-bench/issues/new?template=bench-result.md) and paste the result block

Especially valuable: M1, M1 Pro/Max/Ultra, M2 (all variants), M3 (all variants),
M4 (all variants). We have M5 Max already. The cross-generation trend is the
interesting open question.

## What's in the repo

```
.
├── README.md                 — this file
├── LICENSE                   — Apache 2.0 (covers the code)
├── bench.sh                  — one-command runner
├── requirements.txt          — Python deps
├── bench/
│   ├── convert_bge_coreml.py — one-time mlpackage conversion
│   ├── bench_coreml.py       — CoreML ANE/GPU/CPU bench
│   ├── bench_llama_v2.sh     — llama.cpp Metal bench (parses internal timings)
│   └── bench_mlx.py          — MLX-embeddings bench (with materialization fix)
├── corpus/
│   ├── build_corpus.py       — deterministic corpus build
│   ├── sources.txt           — public-domain text (Pride and Prejudice)
│   └── NOTICE.md             — corpus is PUBLIC DOMAIN, not Apache 2.0
├── docs/
│   └── results-m5-max.md     — canonical M5 Max writeup
├── community-results/        — crowdsourced data points
└── .github/ISSUE_TEMPLATE/
    └── bench-result.md       — issue template for submissions
```

## Methodology

- **Model**: `BAAI/bge-small-en-v1.5` (33M params, BERT-12 encoder, dim=384, FP16). Chosen because it converts cleanly to every backend, is a common production embedder, and is small enough that hardware differences dominate (not model size).
- **Corpus**: 300 sentences from a fixed public-domain text (Pride and Prejudice), bucketed by length: short (~32 tok), medium (~124 tok), long (~462 tok). 100 sentences per bucket.
- **Protocol**: 5 warm runs of 100 sentences per bucket. Drop run 1, take mean of last 4. Cold start timed separately.
- **Throughput unit**: sentences/sec, higher is better. For llama.cpp we use the internal `total_time` metric (not wall clock) because process spawn dominates wall for short benches.
- **Device verification**: CoreML compute-units request is a hint, not a guarantee. We verify actual placement via `MLComputePlan.preferredDeviceForOp`. The reported ANE number is real ANE, not silent CPU fallback.

Full methodology details: [docs/results-m5-max.md](docs/results-m5-max.md).

## Known issues + gotchas

These bit us during development; documented so they don't bite you:

1. **MLX is lazy**. Calling MLX's tensor-materialize function on a model-output wrapper does NOT force compute — you have to materialize the inner tensor (e.g. the `text_embeds` field) explicitly. The bench does this; if you fork it, beware. Otherwise you'll get fake-fast numbers from an unevaluated lazy graph.
2. **coremltools 9 + Python 3.13 + macOS 26 segfaults** in async destructor cleanup. Workaround: `os._exit(0)` after writing results, bypassing the destructor path. Already applied in bench scripts. Apple should fix this; Python 3.12 is unaffected.
3. **llama.cpp wall-time != inference time**. Process spawn / Metal pipeline init can take ~500ms. We use llama's `total_time` print instead. If you compute throughput from wall, you'll see numbers ~10× lower than they should be.
4. **CoreML mlpackage is traced at one fixed shape**. The bundled conversion uses batch=1, seq=512. Re-tracing for different shapes requires running `convert_bge_coreml.py` with modified constants and is its own deployment burden.

## Licensing

- **Code**: Apache 2.0 (see [LICENSE](LICENSE))
- **Corpus** (`corpus/sources.txt`): public domain (see [corpus/NOTICE.md](corpus/NOTICE.md))
- **Documentation**: Apache 2.0

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Most valuable contribution: submit your
bench results from non-M5 hardware via the issue template.
