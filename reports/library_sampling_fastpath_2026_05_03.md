# Library Sampling Fast Path - 2026-05-03

This report measures a small library-facing optimization on
`library-contextbp-backend`: `sample()` and `sample_with_orders()` now generate
directly instead of routing through `sample_with_trace()` and allocating one
`OrderSampleStep` object per generated event.

Raw benchmark outputs are under
`outputs/library_sampling_fastpath_2026_05_03/`.

## Setup

- Baseline commit: `f236cef82e918796120b494a3841b606b51588fb`
- Current change: working tree after the sampling fast path
- Benchmark script: `scripts/eval_library_backend_performance.py`
- Bach pitch-only corpus length: 592 events
- Prefix: first 6 pitch events
- Horizon: 32
- MAXORDER forbidden copied n-grams: 5
- Final pitch class: C
- Samples: 100
- Warmups: 1
- Measured repeats: 7
- Policy: `LongestFeasiblePolicy`
- Execution: single-process CPU Python

## Raw Order-Stack Timing

Sampling medians are reported per generated sequence. BP medians are reported
for the regular masked order-stack BP preparation call.

| K | baseline sample ms/seq | fast-path sample ms/seq | sampling speedup | baseline BP ms | fast-path BP ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1540 | 0.1140 | 1.35x | 18.73 | 19.01 |
| 2 | 0.1263 | 0.0821 | 1.54x | 21.79 | 21.79 |
| 3 | 0.1287 | 0.0837 | 1.54x | 21.57 | 21.32 |
| 4 | 0.1339 | 0.0891 | 1.50x | 21.84 | 21.38 |
| 5 | 0.1410 | 0.0979 | 1.44x | 21.77 | 21.68 |
| 6 | 0.1575 | 0.1072 | 1.47x | 21.77 | 21.95 |

## Public API Timing

Measured at K=6. Prepare time includes each method's existing preparation
scope; sampling time is per generated sequence.

| method | baseline sample ms/seq | fast-path sample ms/seq | sampling speedup | baseline prepare ms | fast-path prepare ms |
|---|---:|---:|---:|---:|---:|
| `raw_prebuilt_model_and_constraint` | 0.1594 | 0.1089 | 1.46x | 22.25 | 21.88 |
| `prepare_constrained_order_stack` | 0.1551 | 0.1151 | 1.35x | 30.71 | 30.90 |
| `prepare_constrained_order_stack_from_events` | 0.1616 | 0.1266 | 1.28x | 42.85 | 43.17 |
| `prepare_continuation_backend` | 0.1567 | 0.1160 | 1.35x | 42.01 | 42.92 |

## Interpretation

- The optimization only affects non-trace sampling APIs.
- `sample_with_trace()` remains available and keeps the richer per-step
  diagnostics.
- Target distributions and constraints are unchanged. Tests compare
  `sample_with_orders()` against `sample_with_trace()` under the same RNG seed
  for both positional and regular prepared backends.
- BP/backward timing is unchanged within benchmark noise, as expected.

## Discarded Experiment

A denser regular-backward cache was also prototyped before this change. On the
same Bach MAXORDER setup it regressed BP timing by roughly 15-25%, so it was
removed rather than committed. The current fast path is therefore focused on
the measured sampling hotspot, not the regular backward pass.
