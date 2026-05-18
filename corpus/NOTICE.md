# Corpus Provenance and Licensing

The text bundled in `corpus/sources.txt` is **not** covered by the repository's
top-level Apache 2.0 LICENSE. The bench code is Apache 2.0; the corpus text has
its own status as described here.

## Bundled source

- **Title**: Pride and Prejudice
- **Author**: Jane Austen
- **First published**: 1813
- **Source**: Project Gutenberg EBook #1342 (https://www.gutenberg.org/ebooks/1342)
- **Excerpt**: First ~80KB of the novel body, with Project Gutenberg's
  header and footer license boilerplate stripped per Gutenberg's own terms.

## Copyright status

Pride and Prejudice is in the **public domain** in the United States and most
other jurisdictions because:
- The author died in 1817 (more than 100 years ago).
- The work was first published in 1813 (more than 95 years ago, predating any
  current copyright term).

It cannot be copyrighted by anyone, including this project.

## Why it's bundled rather than downloaded

A fixed bundled corpus guarantees that benchmark results from different
machines are computed over byte-identical input. A live download from Project
Gutenberg or another source would introduce variance (text revisions, server
errors, region-specific edits) that would invalidate cross-machine
comparisons.

## Project Gutenberg terms

The Project Gutenberg License attaches to the *Gutenberg-published edition*,
specifically the header and footer text. We have stripped that header and
footer, leaving only the public-domain novel body. Per Section 1.E of the
Project Gutenberg License:

> 1.E.2. If an individual Project Gutenberg-tm electronic work is derived
> from texts not protected by U.S. copyright law (does not contain a notice
> indicating that it is posted with permission of the copyright holder), the
> work can be copied and distributed to anyone in the United States without
> paying any fees or charges.

The bundled excerpt qualifies under this clause.

## If you want a different corpus

`corpus/build_corpus.py` reads from `corpus/sources.txt`. To bench with your
own corpus, replace that file (preserving the provenance header) and re-run
the build script. The corpus is bucketed into short/medium/long by character
count, so any English prose source will work.

## What IS covered by Apache 2.0

Everything else in this repository:
- `bench/` scripts
- `corpus/build_corpus.py`
- `bench.sh` and any other tooling
- Documentation files (README, CONTRIBUTING, results writeups)
