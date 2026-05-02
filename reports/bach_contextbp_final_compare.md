# Bach final-C ContextBP comparison

Benchmark generated on 2026-05-02. This comparison intentionally omits
MAXORDER because Continuator's current public `ConstraintProblem` supports
positional constraints, not forbidden-substring regular constraints.

## Setup

- Corpus: Bach Prelude pitch-only sequence, `592` events
- Alphabet size: `25`
- Prefix: `60 64 67 72 76 67`
- Horizon: `n = 32`
- Constraint: final generated pitch class is C
- MAXORDER: omitted for both engines
- K sweep: `1..6`
- Samples: `100` generated sequences per K
- Seed: `0`
- Repeats: one warmup plus five measured repeats, medians reported

## Reproducibility

Command:

```bash
python scripts/eval_bach_contextbp_final_compare.py \
  --orders 1 2 3 4 5 6 \
  --samples 100 \
  --seed 0 \
  --warmups 1 \
  --repeats 5 \
  --output outputs/bach_contextbp_final_compare.csv \
  --raw-output outputs/bach_contextbp_final_compare_raw.csv
```

Validation:

```bash
python -m pytest -q
```

Result:

```text
14 passed in 1.37s
```

Repository metadata:

- `vo_regular_bp` branch: `main`
- `vo_regular_bp` commit: `0c3f0ffced6cfac58b7cadccacf55427bb5763e3`
- `vo_regular_bp` dirty: yes
- Continuator branch: `main`
- Continuator commit: `2bc2d47b90f09cde68f51095fa59bf335a5b35fd`
- Continuator dirty: yes
- Python: `3.13.11`
- NumPy: `2.4.4`
- CPU: Apple M1 Pro, 10 cores, 32 GB RAM
- Threads/processes: single Python process, no multiprocessing

CSV outputs:

```text
outputs/bach_contextbp_final_compare.csv
outputs/bach_contextbp_final_compare_raw.csv
```

## Timing boundaries

For both engines:

- `parse_s`: load the pitch corpus.
- `model_build_s`: learn continuation counts/model.
- `graph_build_s`: compile order graphs for orders `1..K`.
- `constraint_build_s`: build final-C positional constraint/mask.
- `bp_s`: compute backward messages over all compiled order graphs.
- `sampling_per_sequence_s`: sample from the precomputed messages.

This separates BP/message computation from sampling. It does not use
Continuator's public `sample_sequence_with_trace` directly, because that public
method rebuilds order graphs/messages inside each sample call. Instead, the
benchmark uses Continuator's current `ContextBPModel`, graph compiler, and
`backward_messages`, then reuses the messages for 100 samples, matching the
`vo_regular_bp` timing boundary.

## Results

| Engine | K | Model s | Graph s | Constraint s | BP s | Sample s/seq | Context states | Context edges | Reachable states time-indexed | Reachable edges | Violations | Avg copied span | Avg order |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Continuator ContextBP | 1 | 0.000389 | 0.000141 | 0.000028 | 0.000673 | 0.000145 | 27 | 120 | 838 | 3592 | 0 | 6.23 | 1.00 |
| vo-regular-bp policy stack | 1 | 0.000285 | 0.000144 | 0.000000 | 0.000252 | 0.000163 | 25 | 117 | 805 | 3561 | 0 | 6.24 | 1.00 |
| Continuator ContextBP | 2 | 0.000652 | 0.000528 | 0.000029 | 0.002641 | 0.000189 | 174 | 459 | 5350 | 13452 | 0 | 8.84 | 2.00 |
| vo-regular-bp policy stack | 2 | 0.000522 | 0.000579 | 0.000000 | 0.001072 | 0.000211 | 167 | 449 | 5252 | 13361 | 0 | 8.88 | 2.00 |
| Continuator ContextBP | 3 | 0.000924 | 0.001379 | 0.000032 | 0.006658 | 0.000237 | 540 | 1102 | 16090 | 31219 | 0 | 12.31 | 3.00 |
| vo-regular-bp policy stack | 3 | 0.000805 | 0.001547 | 0.000000 | 0.002763 | 0.000257 | 524 | 1080 | 15895 | 31040 | 0 | 12.34 | 3.00 |
| Continuator ContextBP | 4 | 0.001139 | 0.002872 | 0.000030 | 0.013055 | 0.000272 | 1210 | 2102 | 33433 | 55244 | 0 | 16.53 | 4.00 |
| vo-regular-bp policy stack | 4 | 0.001045 | 0.003304 | 0.000000 | 0.005929 | 0.000296 | 1180 | 2062 | 33113 | 54954 | 0 | 16.31 | 4.00 |
| Continuator ContextBP | 5 | 0.001450 | 0.005237 | 0.000033 | 0.022806 | 0.000323 | 2237 | 3502 | 55009 | 81979 | 0 | 20.99 | 5.00 |
| vo-regular-bp policy stack | 5 | 0.001347 | 0.008013 | 0.000000 | 0.010102 | 0.000340 | 2187 | 3437 | 54529 | 81549 | 0 | 20.99 | 5.00 |
| Continuator ContextBP | 6 | 0.001769 | 0.008430 | 0.000035 | 0.034938 | 0.000362 | 3664 | 5344 | 84895 | 117253 | 0 | 20.89 | 6.00 |
| vo-regular-bp policy stack | 6 | 0.001620 | 0.009365 | 0.000000 | 0.015821 | 0.000400 | 3587 | 5246 | 84218 | 116653 | 0 | 20.89 | 6.00 |

Approximate end-to-end median time from parse through 100 samples:

| Engine | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 |
|---|---:|---:|---:|---:|---:|---:|
| Continuator ContextBP | 0.015865 | 0.022934 | 0.032854 | 0.044475 | 0.061977 | 0.081538 |
| vo-regular-bp policy stack | 0.017162 | 0.023523 | 0.031009 | 0.040111 | 0.053669 | 0.067018 |

## Semantics and fairness

The comparison is reasonably fair as a low-level implementation-speed
comparison for positional final-C constraints with longest-feasible order
selection:

- Both engines are pitch-only.
- Both use order graphs for orders `1..K`.
- Both use longest-feasible order selection.
- Singleton avoidance is disabled in this benchmark.
- Both precompute messages once and reuse them for 100 samples.
- Both enforce the final-C constraint with zero violations.

It is not a comparison of the full paper MAXORDER experiment, because MAXORDER
is omitted. It is also not a comparison of Continuator's public sampling call
as shipped, because the public call recomputes graphs/messages per sample; this
benchmark separates graph construction, BP, and sampling to match the
`vo_regular_bp` timing boundaries.

The engines have small semantic/encoding differences:

- Continuator includes hidden START/END symbols internally; `vo_regular_bp`
  policy stack here uses the pitch sequence without boundary tokens.
- Continuator graphs are encoded as integer vocabulary ids; `vo_regular_bp`
  uses pitch values directly.
- Without MAXORDER, highest-order paths remain feasible, so both engines select
  order `K` for every sampled event.

Main observation: `vo_regular_bp` has faster BP/message computation, especially
for larger K, while Continuator has slightly faster graph construction and
slightly faster sampling per sequence in this positional-only setting. End to
end, `vo_regular_bp` becomes faster from K=3 upward under these timing
boundaries.
