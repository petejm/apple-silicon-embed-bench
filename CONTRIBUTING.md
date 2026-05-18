# Contributing

Thanks for your interest! Three ways to help:

## 1. Submit bench results from your hardware (most valuable)

We have M5 Max numbers. We need everything else: M1, M1 Pro/Max/Ultra,
M2 / M2 Pro / Max / Ultra, M3 / M3 Pro / Max / Ultra / M3 Max, M4 / M4 Pro /
Max, etc. Different macOS versions too.

How:
1. `./bench.sh`
2. Wait ~5-15 minutes
3. Copy everything between `===BEGIN RESULT===` and `===END RESULT===`
4. [Open a new issue](https://github.com/petejm/apple-silicon-embed-bench/issues/new?template=bench-result.md) and paste

No need to interpret the numbers; we'll aggregate them. The more submissions
the cleaner the cross-generation trend.

## 2. Fix bench bugs / add backends

PRs welcome for:
- Bug fixes in the bench scripts (wrong timings, parsing errors, etc.)
- Additional backends (PyTorch MPS, Tinygrad, candle-rs, etc.) — see the
  pattern in `bench/bench_mlx.py` and `bench/bench_coreml.py`
- Additional models (anything that converts cleanly to all backends — e.g.
  `nomic-embed-text-v1.5`, `gte-small`, `e5-small`)
- Methodology improvements (better protocol, more rigorous statistics, etc.)
- Documentation improvements

PR checklist:
- [ ] Code changes don't break existing backends
- [ ] If you add a backend, include device-placement verification (compute_units
      requests are HINTS, not guarantees on CoreML)
- [ ] If you add a model, include the conversion script and verify parity
      across backends (cosine sim ≥0.999 between any two backend outputs on
      the same input)
- [ ] Update README results table only if you've actually measured

## 3. Help me reach the llama.cpp maintainers

The bench data shows llama.cpp's BERT-embed Metal path is 4-7× slower than
MLX on the same hardware. That's a real upstream optimization opportunity.
If you have relationships with the llama.cpp project, or experience
optimizing Metal compute shaders, please reach out via an issue.

## Development setup

```bash
git clone https://github.com/petejm/apple-silicon-embed-bench
cd apple-silicon-embed-bench
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 corpus/build_corpus.py
```

For testing changes to a single backend, run that backend directly:

```bash
python3 bench/bench_coreml.py --compute ane --out results/coreml_ane.json
python3 bench/bench_mlx.py
bench/bench_llama_v2.sh
```

## Code style

- Bash: `set -euo pipefail`, quote variables, fail closed on errors.
- Python: standard, follow what's already there. No external linters required.
- Comments: explain *why*, not *what*. Comments noting non-obvious behavior or
  workarounds (e.g. the lazy MLX evaluation gotcha) are very welcome.

## Licensing of contributions

By contributing code or documentation, you agree your contributions are
licensed under Apache 2.0 (matching the project's LICENSE).

Bench result submissions in GitHub issues are CC0 / public domain by default —
they're factual data, not creative works.
