# Library Backend Performance Pass - 2026-05-03

This report measures the current `library-contextbp-backend` branch after the
public backend, event-adapter, Continuator facade, sampling-cache, and
constraint-check hot-loop changes.

Raw outputs are under `outputs/library_backend_perf_2026_05_03/`.

## Environment

- Branch: `library-contextbp-backend`
- Current library commit measured: `0680737`
- Main comparison commit: `670f7fc`
- Previous library comparison commit: `86caa5e`
- Python: `3.13.11 | packaged by Anaconda, Inc.`
- Platform: `macOS-15.7.4-arm64-arm-64bit-Mach-O`
- CPU count reported by Python: `10`
- Execution: single-process CPU Python
- Multiprocessing/vectorized external runtime deps: none

The comparison runs used `VO_REGULAR_BP_IMPORT_ROOT` to import archived copies
of `main` and `86caa5e` while running the same benchmark script.

## Setup

All rows use the Bach Prelude pitch-only setup:

- training length: 592 events
- prefix: first 6 events
- horizon: 32
- MAXORDER forbidden copied n-grams: 5
- final pitch class: C
- samples: 100
- warmups: 1
- measured repeats: 10
- policy: `LongestFeasiblePolicy`

For raw BP rows, the dense MAXORDER acceptor and order-stack model are built
once, graphs are warmed/cached, and timing covers the regular order-stack BP
call plus repeated sampling from a prepared result. It does not include full
model construction.

## Raw Backend Regression

| K | time states | edges | main BP ms | prev BP ms | current BP ms | current vs prev BP | main sample ms | prev sample ms | current sample ms | current vs main sample |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16253 | 74115 | 21.54 | 21.25 | 18.51 | 1.15x | 23.49 | 14.78 | 14.38 | 1.63x |
| 2 | 22059 | 82395 | 24.06 | 24.29 | 21.07 | 1.15x | 30.39 | 12.01 | 11.74 | 2.59x |
| 3 | 24522 | 83494 | 23.77 | 24.66 | 21.78 | 1.13x | 34.36 | 12.41 | 12.78 | 2.69x |
| 4 | 24564 | 83515 | 24.44 | 24.07 | 21.24 | 1.13x | 36.43 | 12.92 | 12.61 | 2.89x |
| 5 | 24586 | 83536 | 24.22 | 24.28 | 21.47 | 1.13x | 36.30 | 13.86 | 13.91 | 2.61x |
| 6 | 24608 | 83557 | 24.06 | 24.38 | 21.36 | 1.14x | 38.66 | 14.73 | 14.20 | 2.72x |

Interpretation:

- The latest hot-loop change improves BP/backward time by about 13-15% versus
  the previous library commit.
- Sampling remains much faster than `main` because of the previously added
  candidate-set cache and cumulative-weight sampler.
- Reachable time-indexed states and product edges are unchanged across the
  compared commits for the same K, so this is an implementation-speed change,
  not a target-distribution change.

## Public API Overhead

Measured at K=6.

| method | prepare median ms | sample median ms / 100 | sample us / seq |
|---|---:|---:|---:|
| `raw_prebuilt_model_and_constraint` | 21.49 | 13.82 | 138.2 |
| `prepare_constrained_order_stack` | 30.56 | 14.64 | 146.4 |
| `prepare_constrained_order_stack_from_events` | 41.98 | 15.34 | 153.4 |
| `prepare_continuation_backend` | 41.97 | 15.48 | 154.8 |

Timing scope:

- `raw_prebuilt_model_and_constraint`: model graphs and dense acceptor already
  exist; timing is essentially the BP preparation call.
- `prepare_constrained_order_stack`: caller supplies a model; timing includes
  constraint compilation and backend preparation, with the supplied model reused
  across repeats.
- `prepare_constrained_order_stack_from_events`: timing includes event encoding,
  model construction, constraint compilation, and backend preparation.
- `prepare_continuation_backend`: same broad scope as the event API, plus the
  Continuator-shaped facade logic.

The nicer APIs add modest overhead during preparation. Sampling overhead through
the event/Continuator facades is small: roughly 138 us/sequence raw versus
155 us/sequence through `prepare_continuation_backend` on this setup.

## Profile Summary

Profile files:

- `outputs/library_backend_perf_2026_05_03/current_profile_bp.txt`
- `outputs/library_backend_perf_2026_05_03/current_profile_sampling.txt`

K=6 BP/backward profile over 20 preparations:

- `_RegularBackwardCache.beta` dominates nearly all cumulative time.
- `dict.get` inside beta memoization is the main visible sub-cost.
- Graph compilation is not visible in the hot profile because graphs are cached
  before timed BP, matching the paper timing style.

Top BP profile lines:

```text
1649940/120 calls  order_stack_bp.py:632(beta)        1.548 s cumulative
3993440 calls      {method 'get' of 'dict' objects}  0.424 s cumulative
```

K=6 sampling profile over 20 x 100 samples:

- `sample_with_trace`, policy `choose`, and `_candidate_sets` dominate.
- `_sample_from_weighted_edges` is now a smaller share.
- Candidate-set caching is working, but constructing traces and policy
  decisions remains visible.

Top sampling profile lines:

```text
64000 calls  order_stack_bp.py:84(choose)            0.344 s cumulative
64000 calls  order_stack_bp.py:850(_candidate_sets) 0.192 s cumulative
64000 calls  order_stack_bp.py:928(_sample_from_weighted_edges) 0.062 s cumulative
```

## Next Optimization Candidate

The next substantial speedup should target `_RegularBackwardCache.beta`, not
graph construction. The profile suggests:

1. Replace recursive dict-key beta memoization with a denser iterative dynamic
   program for dense integer DFA states.
2. Specialize the dense MAXORDER regular backend so keys become compact integer
   indices rather than `(time, context_state, acceptor_state)` tuples.
3. Keep the current generic DFA path available for arbitrary acceptors.

The array-backed edge experiment was tried during this pass and discarded: it
made BP slower on the Bach MAXORDER benchmark, so it was not committed.
