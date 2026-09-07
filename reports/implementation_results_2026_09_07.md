# Implementation results — 7 September 2026

Developed on `codex/bp-correctness-performance`, compared with baseline `4f7c871`.
The Python library remains dependency-free and existing public call signatures
remain compatible. This report covers the implementation and validation being
integrated into `main`; it does not introduce a new package version or release tag.

**Completed work**

- Lazy virtual graphs preserve terminal emissions and resolve previously unseen
  backoff contexts against exact reverse reachability. Small randomized corpora
  compare their complete support and outgoing rows with materialized graphs.
- Padded-duration specialization validates every equal-cost symbol and the
  acceptance, padding, final-note, and note-count behavior it replaces. The
  comparison test now prepares the actual optimized backend and asserts that
  specialization is active. Nonconforming DFAs retain the generic path.
- Regular transition-row caches retain at most 65,536 entry units per order;
  full caches stop admitting new rows rather than churning through evictions.
  DFA-symbol transitions are shared across orders within a plan, while dynamic
  soft acceptors receive independent caches. Diagnostics expose retained entries
  and skipped admissions.
- Virtual source graphs are shared across prefixes/horizons. Explicit cache
  release leaves existing plans usable. Generic product BP shares static
  transition rows between time layers.
- Sequence-only calls avoid discarded order arrays. The built-in longest-feasible
  policy avoids decision metadata, cached-list copies, and unnecessary lower-order
  evaluations when traces are not requested. Sampling reuses cached DFA transitions.
  Same-seed sequence/order/trace equivalence is tested, including singleton policy.
- First-hit generation without extra constraints shares suffix views of one
  backward table instead of constructing a table per length. Additional
  constraints retain the general implementation and its existing semantics.
- Product and positional engines expose logarithmic partition/conditional
  probabilities. Order-stack results expose logarithmic start masses. Stable
  fallbacks preserve sampling when float masses underflow or overflow, and
  first-hit lengths use proportional weights when absolute masses cannot be
  represented. Positional traversal is iterative; long regular horizons use an
  iterative log evaluator. Short ordinary regular queries retain the fast kernel.

**Verification**

The final suite has 108 passing tests, up from 83. Coverage includes explicit
versus virtual support, specialized versus generic distributions, graph reuse,
cache budgets, dynamic-soft-cache isolation, seeded sampling/trace equivalence,
1,200-symbol horizons, zero support, extreme finite soft weights, first-hit length
ratios beyond float range, and empty-horizon diagnostics. Ruff's F checks pass
on changed Python files, and `git diff --check` passes. Repository-wide F checks
also identify one pre-existing unused local in `orbit_diagnostics.py`.
Formatting of changed regions was checked to preserve
the parsed Python AST. Tests were run on Python 3.13.11; the full CI Python matrix
was not executed locally.

**Measurement method**

`scripts/bench_backend_lifecycle.py` runs each repetition in a fresh process and
accepts `--source-root` to run the same driver against a saved baseline checkout.
It separates preparation, first sample, subsequent samples, repeated warm samples,
first trace batch, repeated warm traces, and peak RSS. An optional separate
`--memory` run records traced Python allocations. Workers enforce configurable
time and RSS limits; these resource checks use Unix APIs.

Ordinary timing results below are medians of three runs on the same macOS arm64
machine, Python 3.13.11. Memory tracing is a separate single run and its timings
are not used for CPU claims. All compared benchmark start masses match within
`1e-10` relative tolerance; recorded first samples match exactly. These are local
measurements rather than hardware-independent guarantees.

Raw runs, parameters, environment, and medians:
[`performance_results_2026_09_07.json`](performance_results_2026_09_07.json).

| Workload and operation | Baseline | Implemented | Change |
|---|---:|---:|---:|
| Full weighted LSDB preparation | 2.277 s | 2.007 s | 12% less time |
| Full weighted LSDB preparation, peak RSS | 333.0 MiB | 217.6 MiB | 35% less |
| Full weighted LSDB preparation, retained traced allocations | 285.1 MiB | 170.8 MiB | 40% less |
| Smaller LSDB, first sample | 495 ms | 390 ms | 21% less time |
| Smaller LSDB, next 20 samples | 645 ms | 518 ms | 20% less time |
| Smaller LSDB, repeat same 20 samples with warm caches | 0.995 ms | 0.414 ms | 58% less time |
| Smaller LSDB, peak lifecycle RSS | 242.7 MiB | 161.5 MiB | 33% less |
| Bach, next 1,000 samples | 114.7 ms | 46.6 ms | 59% less time |
| Bach, repeat same 1,000 samples with warm caches | 74.5 ms | 19.4 ms | 74% less time |
| First-hit lengths 1–1,024, preparation | 879 ms | 6.53 ms | 99.3% less time |
| First-hit lengths 1–1,024, peak lifecycle RSS | 254.3 MiB | 25.4 MiB | 90% less |

Full LSDB uses 120 tracks × 96 events, 12 virtual transpositions, maximum order 4,
horizon 32, total duration 16, and the weighted duration/motion DFA. Smaller LSDB
uses 16 tracks, order 3, and horizon 16. Bach uses order 6, horizon 32, forbidden
copied 5-grams, and final pitch class C. These fixed-horizon benchmarks use
`LongestFeasiblePolicy`. The first-hit benchmark is a two-state model; its large
gain demonstrates removal of quadratic horizon duplication and should not be
extrapolated to arbitrary additional constraints.

**Tradeoffs and remaining costs**

- The production preparation gain is approximately 12%, below the initial
  25–30% target suggested by discarding all transition rows. Bounded retention
  preserves reuse for smaller recurrent products, and correctness/numerical
  checks carry overhead. The memory target was met on the measured workload.
- Bach preparation increases from 45.9 to 50.9 ms (about 5 ms / 11%). Peak Bach
  lifecycle RSS increases from 37.4 to 38.9 MiB. Repeated sequence generation is
  substantially faster; preparation-only applications do not receive that gain.
- The first trace batch can be slower because plain sampling now avoids work
  needed only to report every candidate order. Smaller LSDB's first 20 traces
  take 145 rather than 70 ms; Bach's first 1,000 traces take 169 rather than
  128 ms. Fully warm trace batches are much closer: 1.52 versus 1.32 ms for
  smaller LSDB and 118 versus 115 ms for Bach. Cached plain/trace data is separate
  so diagnostic completeness does not change the fast policy.
- The full LSDB first sample still needs substantial lazy expansion: a separate
  implementation-only run took 13.39 s, and the lifecycle through additional
  plain and trace samples peaked at about 1.46 GiB RSS. That observation is not a
  controlled speedup comparison. The largest workload still warrants future
  profiling of reachable products and symbol-cache storage.
- Numerical fallback incurs extra work only when needed for extreme messages
  or long regular horizons. It cannot recover a weight already rounded to zero
  inside a caller's weight function, and individual returned weights must remain
  finite. Ordinary float partition properties may still be zero/infinite outside
  their representable range; logarithmic properties carry the useful value.
- Shared first-hit tables currently require `constraints=None`. Supporting
  additional time-homogeneous constraints is a possible extension. General
  horizon-dependent constraints still require independent computation.

**Reproduction**

Run the same commands with `--source-root /path/to/baseline` for the baseline.
The saved baseline used here was an archive of commit `4f7c871`.

```sh
python scripts/bench_backend_lifecycle.py --case lsdb --tracks 120 --length 32 --max-order 4 --samples 0 --repeats 3
python scripts/bench_backend_lifecycle.py --case lsdb --tracks 120 --length 32 --max-order 4 --samples 0 --memory --repeats 1
python scripts/bench_backend_lifecycle.py --case lsdb --repeats 3
python scripts/bench_backend_lifecycle.py --case bach --max-order 6 --length 32 --samples 1000 --repeats 3
python scripts/bench_backend_lifecycle.py --case until --length 1024 --samples 20 --repeats 3
python -m pytest -q
```

Further compiled acceleration, source suffix-DAG redesign, and finite-horizon
product quotienting remain optional architectural projects. The implemented
changes establish correctness coverage and lifecycle measurements to assess
those projects without relying on preparation-only timings.
