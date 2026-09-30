# MaxOrder (2014): reproducible Onegin reconstruction

This experiment accompanies the revised MaxOrder section of *Markov Constraints
and Controlled Sequence Generation*. It revisits Papadopoulos, Roy and Pachet's
2014 MaxOrder experiment using `vo-regular-bp` and explicit corpus, boundary,
startup and backoff conventions.

The reconstruction recovers **29** strict-order-3, length-20 sequences avoiding
copied nine-word fragments. Their raw mass is **1.0001778490e-22** under order-1
scoring and **4.8815273941e-5** under order-3 scoring. This supports a distinction
between feasibility and scoring order; it does not establish the historical
implementation. The historical copy-length quartiles remain unreproduced.

## Contents

- [Complete results and protocol](reference/README.md).
- `prepare_corpus.py`: extract three corpus/boundary variants and verify the
  cleaned source hash. The full translation is downloaded separately.
- `reproduce.py`: strict/backoff sources, baseline sampling and exact support counts.
- `verify_counts.py`: independent integer/rational forward DP without library imports.
- `score_witnesses.py`: independent enumeration and rational rescoring of all 29 witnesses.
- `run_experiment.py`: reproduction, reference-integrity checks and output comparison.
- `generate_artifacts.py`: vector PDF curve and LaTeX copy-length table.
- `reference/`: unchanged September 30 evidence, 36,000 baseline samples, exact
  verification records, original script snapshots and SHA-256 manifest. Paths and
  PDF metadata inside these records describe the original manuscript run.

## Reproduce

Run from the repository root with Python 3.10 or later. Core computation uses
only the standard library and this checkout; PDF generation additionally needs
ReportLab. The recorded numerical baseline uses library commit
`614697b3e16444dc7ca81d4f74fe273b7d3b28a0`. The runner records the actual library
commit and Python module hashes, and compares scientific outputs with the archive.

```sh
# Optional: needed only for --figures or generate_artifacts.py.
python3 -m pip install -r experiments/maxorder_2014/requirements-figures.txt

# Hash verification alone requires no corpus or optional packages.
python3 experiments/maxorder_2014/run_experiment.py --verify-only

# Main results: all 17 thresholds, 12 sampling conditions, independent checks.
# Choose a fresh output directory; reference evidence is never overwritten.
python3 experiments/maxorder_2014/run_experiment.py --download \
  --output /tmp/maxorder-reproduction --figures
```

Use `--source /path/to/onegin.html` instead of `--download` for a saved source.
Add `--full` to reproduce all boundary and horizon comparisons (36,000 baseline
samples in total). `--library /path/to/checkout` selects a different or pinned
library checkout. Counts, rational witness scores, sampled sequences and summary
statistics are checked against the archived results; timing is not an invariant.
The source extraction fails if the cleaned chapter-body hash changes.

Regenerate the chapter's figure and table directly from verified reference data:

```sh
python3 experiments/maxorder_2014/generate_artifacts.py \
  --output /tmp/maxorder-reference-figures
```

A small exhaustive check without downloading Onegin:

```sh
python3 experiments/maxorder_2014/reproduce.py --library . \
  --self-test --output /tmp/maxorder-toy-check
```

The original scripts and manuscript-specific paths in `reference/` are historical
snapshots. Use the package-level commands above for portable reproduction.

## Package validation

The [full package validation](validation.json) reran all eight experiment groups
with Python 3.12.14 and the recorded engine revision. All 36,000 baseline
sample sequences, exact witness scores, and reference statistics matched. The
regenerated PDF curve and LaTeX table were byte-identical to the manuscript
artifacts. This check used the saved source HTML; the optional downloader was
not exercised.

A [pre-publication check](publication_validation.json) also ran the main protocol
with Python 3.14.6 and a fresh source download. All 12,000 baseline samples, all
17 exact count/probability checks, and all 29 witnesses and rational scores
matched the archive. The download and cleaned-source hash check passed.

## Related implementations

- [vo-regular-bp](https://github.com/fpachet/vo-regular-bp): regular-constrained
  counting, weighted inference and sampling used here.
- [Snarky Markov Constraints application](https://github.com/fpachet/snarky/blob/main/docs/markov_constraints_application.md):
  reconstruction of the 2011 optimization work, including melody controls.
- [Ordinary, exotic and Boulez Blues evidence](https://github.com/fpachet/snarky/tree/main/docs/research/blues_villani_2026-09-16):
  reconstructed corpus, exact scores, optimality evidence and source archives.

Keep this package with the library and cite a fixed release or commit when
publishing subsequent results. The baseline library commit predates addition of
this experiment directory; it identifies the engine, not this package's release.
