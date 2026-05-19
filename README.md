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

How fast can each backend answer a one-sentence query? Relevant for interactive search, RAG query embedding. Headline numbers are 10-run means from 10 independent `./bench.sh` invocations; the bundled per-run JSON in `community-results/<chip>/` is run 1 of N.

| Hardware | CoreML ANE | CoreML GPU | CoreML CPU | MLX |
|---|---:|---:|---:|---:|
| M5 Max | 89 | 572 | 59 | 562 |
| M4 Pro | 80 | 173 | 73 | 384 |

Sentences/sec on short bucket (~32 tok). CoreML rows pad every input to seq=512 (the mlpackage is traced at that fixed shape). MLX b=1 is a 100-sentence loop with one inference call per sentence.

### Throughput (natural batching)

How fast can each backend chew through a corpus? Relevant for indexing, bulk reindex. Headline numbers are 10-run means from 10 independent `./bench.sh` invocations; the bundled per-run JSON in `community-results/<chip>/` is run 1 of N.

| Hardware | llama.cpp Metal (n_seq~66 batched) | MLX (100-in-one-call) |
|---|---:|---:|
| M5 Max | **1,176** | **3,099** |
| M4 Pro | 933 | 869 |

Sentences/sec on short bucket (~32 tokens). llama.cpp does internal batching (~66 seq per forward pass). MLX numbers are an extreme batched case (100 in one call); a more realistic b=16 or b=32 would land lower.

Full 10-run variance tables (min/max/σ/CV per cell): [docs/results-m5-max.md](docs/results-m5-max.md) + [community-results/](community-results/).

> **Correction note**: an earlier version of this README headline showed MLX
> batched-short on M5 Max as 6,950 sent/s. A 10-run variance sweep on the
> same machine could not reproduce that number (mean 3,099, range
> 3,070-3,125, σ=18, CV 0.6%). The 6,950 was a measurement artifact and has
> been retracted.

### Headline findings (with appropriate caveats)

1. **ANE-via-naive-FP16-coremltools-trace is the slowest GPU-class path** on both M4 Pro and M5 Max for this model. Not the fastest. (Different conversion paths, especially INT8 + `ml-ane-transformers`, may differ; not tested here.)
2. **ANE perf is roughly flat across M-generations** (~80-90 sent/s on both M4 Pro and M5 Max). GPU is where Apple is shipping perf gains: 3.3× CoreML GPU and 3.6× batched MLX from M4 Pro → M5 Max. **Form-factor-honest framing**: this is M5 Max (MacBook, 64 GB) vs M4 Pro (Mac mini, 24 GB) — a single-pair comparison across two different thermal envelopes. Thermal headroom is a confound, not just silicon generation. We do NOT claim "across M-generations" as a silicon-only result.
3. **MLX dominates large-batch throughput** on M5 Max (2.6× faster than llama.cpp Metal at 100-in-one-call on the short bucket; 3.3× on medium, 1.6× on long — gap is largest at medium, not short). The gap shrinks on M4 Pro and at more realistic batch sizes.
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

`bench.sh` runs `./check.sh` first — a pre-flight check that surfaces any missing/incompatible prereqs (macOS version, arm64, Python 3.10-3.13, brew, llama.cpp, disk space, network) up front, before any time is spent on venv setup or downloads. To run the check alone: `./check.sh`. To bypass: `SKIP_CHECK=1 ./bench.sh`.

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
├── check.sh                   — pre-flight environment check (runs automatically)
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
- **Protocol** (two-level):
  - **Inner loop**: a single `./bench.sh` invocation runs 5 warm runs × 100 sentences per bucket, drops run 1 (warmup), and takes the mean of runs 2–5. This produces one point estimate per bucket per invocation. A single inner run is **noisy** — not the canonical number.
  - **Outer loop**: 10 independent `./bench.sh` invocations. The headline cells in the tables above are the **mean of those 10 outer-run means**, reported as `mean [min…max] σ CV%` (see [docs/results-m5-max.md](docs/results-m5-max.md) for the full variance table). Variance characterization requires the outer loop; a single invocation does not.
- **Throughput unit**: sentences/sec, higher is better. For llama.cpp we use the internal `total_time` metric (not wall clock) because process spawn dominates wall for short benches. CoreML pads to seq=512 so effective tok/s differs from sent/s.
- **Device verification**: `compute_units` is a hint, not a guarantee. `probe_devices.py` enumerates actual placement via `MLComputePlan` and writes `results/devices_<unit>.json`; the bench includes this in the result block.

### Reproducing the canonical numbers

`./bench.sh` runs one inner-loop invocation — useful for a quick read but **not** the canonical headline. To reproduce the 10-outer-run means in this README, run:

```bash
N_SWEEPS=10 ./bench-sweep.sh
```

`bench-sweep.sh` wraps `bench.sh` in an outer loop, writing each run's `results/` to `results/run-NN/`, then prints an aggregate summary (`results/sweep_summary.{json,md}` — 10-run mean / range / σ / CV per cell). Total time scales linearly — expect ~50–150 minutes for `N_SWEEPS=10`. The default `./bench.sh` behavior (single invocation, 5 inner runs) is unchanged.

Full methodology details: [docs/results-m5-max.md](docs/results-m5-max.md).

## Known issues + gotchas

These bit us during development; documented so they don't bite you:

1. **MLX is lazy**. Calling MLX's tensor-materialize function on a model-output wrapper (BaseModelOutput) does NOT force compute. Materialize the inner tensor (e.g. `.text_embeds`). The bench does this; if you fork it, beware. First MLX bench reported 43K sent/s; the 10-run reproducible number is ~3,099 sent/s once the inner tensor is forced.
2. **coremltools 9 + Python 3.13 + macOS 26 segfaults** in async destructor cleanup. Workaround: `os._exit(0)` after writing results. Applied in bench scripts. Apple should fix this; Python 3.12 is unaffected.
3. **llama.cpp wall-time != inference time**. Process spawn / Metal pipeline init can take ~500ms. We use llama's `total_time` print instead.
4. **CoreML mlpackage is traced at one fixed shape**. The bundled conversion uses batch=1, seq=512. Re-tracing requires modifying `convert_bge_coreml.py` and may fail on non-M5 silicon. Use `REBUILD_MLPACKAGE=1` only if you want to retrace.
5. **`has tensor = false` in llama.cpp Metal init** on macOS 26.5 + brew build 9150. Apple's M5 tensor accelerators are temporarily disabled in this build. MLX uses them, llama.cpp doesn't — part of the headline MLX-vs-llama gap is this API-state, not closeable code.

## Limitations

Consolidated caveats — read these before generalizing any number in this README:

- **Single conversion path tested**: HF → coremltools 9 → `torch.jit.trace` → FP16 mlpackage at batch=1/seq=512. Apple's `ml-ane-transformers` attention rewrite and INT8 quantization paths are documented ANE optima; neither is tested here. ANE numbers reflect this single conversion path, not ANE-in-general.
- **Single model size**: bge-small-en-v1.5 only (33M params, BERT-12). Findings may not transfer to BGE-base, BGE-large, E5-mistral, or larger models where memory bandwidth and model size dominate differently.
- **n=2 hardware variants**: M5 Max (MacBook, 64 GB) and M4 Pro (Mac mini, 24 GB). No M1, M2, M3, base M4, M-Ultra, or M-Max-non-MacBook data points. Cross-generation claims are based on one M5/M4 pair across different form factors.
- **Cross-backend cosine parity check is bundled** (`bench/verify_parity.py`, invoked by `bench.sh`). If any backend fails the >= 0.999 cosine threshold against another, the result block surfaces it — but the four backends could still be doing subtly different things (different pooling, normalization, attention impl). The parity check is a sanity floor, not a proof of equivalence.
- **MacBook thermals are not controlled to steady-state**. The M5 Max numbers are from a MacBook running browser / IDE / window manager — CV is 7.4–7.6% on CoreML GPU at medium/long buckets. The M4 Pro Mac mini variance is much tighter (CV < 0.5%) because of headless / cooled chassis. Form factor matters; we do not claim silicon-only deltas.
- **No INT8 / `ml-ane-transformers` paths tested**. These are the documented ANE optima and would likely close (or reverse) the ANE-vs-GPU gap; we did not test them and our numbers don't speak to them.
- **No dynamic-shape mlpackage tested**. CoreML is traced at fixed batch=1/seq=512. Variable-shape mlpackages punt to GPU on Apple silicon and have their own perf profile that this bench does not measure.

## Licensing

- **Code**: Apache 2.0 (see [LICENSE](LICENSE))
- **Corpus** (`corpus/sources.txt`): public domain (see [corpus/NOTICE.md](corpus/NOTICE.md))
- **Documentation**: Apache 2.0

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Most valuable contribution: submit your
bench results from non-M5 hardware via the issue template.
