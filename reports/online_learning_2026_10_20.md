# Online learning followed by constrained generation — 2026-10-20

## Decision and implementation

Keep the existing inference algorithms and recompute backward messages after
learning. Add immutable, incremental count snapshots and reuse unaffected
ordinary source graph rows. This reduces work before BP without introducing a
mutable model/inference invalidation protocol. Fully incremental BP is not
justified by the measured dependency propagation on the Bach workload.

The additive API is:

```python
model = model.with_sequences([new_phrase])
```

`OrderStackModel.with_sequences` counts only the new independent phrases, copies
the count dictionary and changed `Counter` rows, and remembers START/END padding
from `from_sequences`. Duplicates contribute again. Unchanged count rows and
compiled graph rows are shared read-only. It does not store phrase history.

Each ordinary graph is updated lazily on the next preparation. Its context-ID
containers are copied, changed effective suffix rows are replaced, and new
counted contexts plus their transition closure are added. A longer unchanged
counted context shields its row from shorter-suffix changes. Importantly, a newly
learned longer suffix can replace a shorter backoff distribution and **remove**
transitions even though learning only adds counts. If that happens, the order
falls back to full construction, removing potentially orphaned closure states.
Keeping all old states would change lookup/support for certain prefixes; a
focused regression exercises this case.

This is a small source-graph optimization, not an incremental automaton or BP
implementation. No reverse dependency index, graph tombstones, or historical
inference snapshots are retained. Pending updates coalesce changed-context keys;
there is at most one seed graph per prepared order. Compiling consumes that seed.
Minimized graphs are rebuilt. Ordinary count-dictionary copying and scanning graph
contexts still cost O(number of contexts), and broad updates can trigger the
full-graph fallback.

`VirtualAugmentedOrderStackModel.with_sequences` incrementally updates base
counts but creates a fresh wrapper, rebuilding all transformed views and lazy
graphs. Partial transformed-graph reuse did not have a sufficient measured
justification to add its much broader invalidation machinery.

`with_sequences(..., reuse_graphs=False)` retains count updates but forces full
graph construction. `from_sequences(all_phrase_memory, ...)` remains the
independent count/graph reference. Existing generation signatures and defaults
are unchanged. PR #1's numeric positional recurrence, temporary row-cache bound
and log/underflow fallback are untouched. There are no new runtime dependencies;
all automata and BP remain in this repository.

## Method

Base reference: merged PR #1, commit
`361d077ce605df5ba14147ab303cf4bd32f2efb3`. Ordinary timings use CPython 3.13.11
(Anaconda build), macOS 15.7.7, arm64, in separate worker processes. This is a
local synthetic/library benchmark, not a measurement inside Continuator and not
a universal performance target. The 549-note input is the first 549 pitch tokens
of the repository's Bach corpus, not necessarily the user's MIDI material.

The driver measures a cold initial model and eight subsequent 16-note additions.
Known slices exercise repeated context learning; additions 4 and 8 introduce new
symbols. A separate `novel-phrases` workload generates independent random phrases
on the existing alphabet, also introducing new symbols at those updates. Synthetic
corpora use a fixed random seed and 12 pitches. Orders 1, 2, 4 and 6, horizons
16, 32, 64 and 128, and corpora of 64, 549, 4096 and 16384 notes are covered.

Methods: `rebuild` reconstructs all counts from phrase memory; `counts` updates
counts but reconstructs graphs; `incremental` also reuses ordinary graph rows.
Virtual `counts` and `incremental` deliberately use the same implementation.
Five isolated workers per method/case alternate method execution order. Reported
post-learning values are the median of each worker's eight-cycle median; cold
values are medians of the five rebuild workers. Raw per-cycle data and worker
ranges are retained. Components are independently summarized, so rounded component
medians need not sum to the median total.

Fixed cases force a final C pitch available in the corpus. Regular cases also
forbid `(60, 60, 60)` through the normal compiled DFA. Until-END learns END,
forbids it before the last emission, and selects lengths in `[1, 64]` using the
existing first-hit weighting. Virtual cases use eight shifts, `range(8)`, with
absolute overlapping contexts. Virtual regular preparation uses lazy graphs;
its backward/preparation stage includes source-row expansion, which cannot be
honestly attributed to the near-zero graph-container setup time.

Stages explicitly include dropping the old prepared backend, storing the phrase,
count construction/update and old model disposal, graph setup, backward/preparation,
the first sample, and ten subsequent samples sharing inference. The total ends
after the first sample. Source graph setup is called separately to expose its
cost; preparation then uses those same graph caches. Backward/preparation includes
constraint compilation, masks and prefix binding. Sampling can expand regular
caches. Input generation, imports, persistence, event/MIDI encoding, UI work and
external decoder/cache management are excluded. Corpus storage here is a tuple
append, not Continuator's full phrase storage operation.

Memory runs are separate from ordinary timing runs. `tracemalloc` starts before
initial model construction; each cycle records live allocations after GC and
the cumulative traced peak. Input preparation and imports precede tracing.
Process peak RSS includes the interpreter, imports and allocator arenas and is
reported separately. Timing fields in allocation-traced runs are not used for
speed comparisons. Old application backends are released each cycle; intentionally
retaining old plans would retain their old snapshots and inference.

## Reproduce

From the repository root:

```bash
python -m pytest -q
python examples/online_learning_backend.py
python scripts/bench_online_learning.py --repeats 5 --updates 8 > timings.jsonl
python scripts/bench_online_learning.py --memory --repeats 1 --updates 8 > memory.jsonl
python scripts/bench_online_learning.py --case bach-o2 --memory --repeats 1 --updates 64 > memory_stress.jsonl
python scripts/diagnose_online_bp.py > dependencies.jsonl
python scripts/summarize_online_learning.py . > summary.json
```

Put the JSONL files in one directory before calling the summary script. Individual
cases/methods are selectable with `--case` / `--method`. `--source-root` can point
to another checkout, using `--method rebuild` for versions without the new API.
Use `--case bach549 --method incremental --repeats 1 --profile /tmp/online.prof`
for cProfile output; profiled timings are not ordinary timing measurements.
Raw checked-in results are in [online_learning_2026_10_20/](online_learning_2026_10_20/).

## Timing results

All times below are milliseconds. Cold totals build the initial model; post-learning totals include invalidation, storage, counts, graphs, preparation and the first sample. Cached sampling is per sequence.

| Case (notes / order / horizon) | Cold total | Rebuild post | Counts-only post | Reuse post | Lower latency | Cached reuse |
|---|---:|---:|---:|---:|---:|---:|
| tiny (64 / 1 / 16) | 0.291 | 0.254 | 0.214 | 0.199 | 21.6% | 0.022 |
| bach549 (549 / 4 / 64) | 14.921 | 15.283 | 14.271 | 11.548 | 24.4% | 0.086 |
| bach-o2 (549 / 2 / 16) | 1.710 | 1.693 | 1.211 | 0.788 | 53.4% | 0.024 |
| bach-o6 (549 / 6 / 128) | 70.003 | 71.158 | 69.387 | 62.942 | 11.5% | 0.261 |
| growing4k (4096 / 4 / 64) | 115.564 | 118.797 | 110.471 | 88.706 | 25.3% | 0.091 |
| growing16k (16384 / 4 / 64) | 306.412 | 313.658 | 277.245 | 215.117 | 31.4% | 0.171 |
| novel-phrases (4096 / 4 / 64) | 118.479 | 124.400 | 115.001 | 91.895 | 26.1% | 0.113 |
| regular (549 / 4 / 64) | 39.357 | 38.627 | 37.896 | 35.227 | 8.8% | 0.120 |
| until-end (549 / 4 / 64) | 15.781 | 16.636 | 15.986 | 12.759 | 23.3% | 0.079 |
| virtual (549 / 4 / 32) | 83.970 | 91.454 | 90.704 | 89.429 | 2.2% | 0.048 |
| virtual-regular (549 / 4 / 16) | 48.437 | 52.475 | 50.623 | 50.516 | 3.7% | 0.026 |

For the 549-note, order-4, 64-note fixed-final case:

| Phase / method | Invalidate + store | Counts | Graphs | Backward / prepare | First sample | Cached sample | Total through first |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold reference | 0.000 | 0.978 | 2.996 | 10.855 | 0.169 | 0.094 | 14.921 |
| Post / rebuild | 0.435 | 1.145 | 2.918 | 10.580 | 0.148 | 0.089 | 15.283 |
| Post / counts only | 0.432 | 0.163 | 2.878 | 10.546 | 0.151 | 0.088 | 14.271 |
| Post / reuse | 0.424 | 0.088 | 0.467 | 10.378 | 0.142 | 0.086 | 11.548 |

The five worker medians for post-learning totals range from 15.139–15.928 ms for rebuilding and 11.534–11.650 ms for reuse. The measured 24.4% reduction comes mainly from graph work (2.918 → 0.467 ms), with count construction also reduced (1.145 → 0.088 ms). Counts-only updating saves much less. Backward work is effectively unchanged; cached sampling is also unchanged within workload/noise differences. Cold generation has no update history to exploit.

The whole-lifecycle cProfile confirms the same separation: nine cold/post-learning preparations spend 0.117 s in graph compilation for rebuilding versus 0.026 s with reuse; the positional backward routine takes 0.117 s versus 0.115 s. Ninety-nine samples take about 0.040 s versus 0.041 s. These instrumented values include one cold cycle and eight updates and are not substituted for uninstrumented timings; imports are visible separately in the [profile](online_learning_2026_10_20/profile.txt).

The novel-phrase case does not erase the ordinary-graph benefit in this matrix. This is not a guarantee for arbitrary topology changes: support-shrink fallback is covered by correctness tests, and its detection can add overhead before a full rebuild. Virtual augmentation shows only a small total change (roughly 2–4%), with noise of similar practical scale and no transformed graph/BP reuse. No virtual speedup is promised.

## Memory results

MiB below are allocation-traced cold retained, end retained after eight updates, cumulative traced peak, and process peak RSS. Pairs are **rebuild / reuse**. Each memory row is one separate worker per method, not a confidence interval.

| Case | Cold retained | End retained | Traced peak | Peak RSS |
|---|---:|---:|---:|---:|
| tiny | 0.101 / 0.101 | 0.106 / 0.107 | 0.109 / 0.110 | 24.4 / 24.3 |
| bach549 | 3.082 / 3.082 | 3.099 / 3.100 | 3.165 / 3.137 | 59.9 / 52.4 |
| bach-o2 | 0.276 / 0.276 | 0.294 / 0.294 | 0.303 / 0.302 | 25.7 / 25.0 |
| bach-o6 | 15.457 / 15.457 | 15.502 / 15.509 | 15.624 / 15.565 | 161.4 / 162.3 |
| growing4k | 19.661 / 19.661 | 19.629 / 19.639 | 20.803 / 20.633 | 177.5 / 166.3 |
| growing16k | 43.250 / 43.250 | 43.259 / 43.286 | 46.497 / 46.301 | 260.3 / 229.5 |
| novel-phrases | 19.661 / 19.661 | 20.268 / 20.284 | 21.426 / 21.202 | 157.5 / 163.9 |
| regular | 5.047 / 5.047 | 5.084 / 5.085 | 5.253 / 5.243 | 51.5 / 56.4 |
| until-end | 3.119 / 3.119 | 3.181 / 3.183 | 3.317 / 3.302 | 58.9 / 56.2 |
| virtual | 15.797 / 15.797 | 16.686 / 16.686 | 17.582 / 17.582 | 135.7 / 147.0 |
| virtual-regular | 7.866 / 7.866 | 9.293 / 9.292 | 15.303 / 15.131 | 62.5 / 63.5 |

Retained allocations are essentially equal; this change is primarily a latency optimization. Peak traced allocation is slightly lower in most ordinary cases, but there is no substantial memory-reduction claim. Single-process RSS peaks vary with allocator behavior and should not be interpreted as reliable comparative wins.

A separate 64-addition stress run at order 2 / horizon 16 retains 0.358 MiB rebuilding versus 0.359 MiB updating (both start at 0.276 MiB); traced peaks are 0.383 / 0.377 MiB. This run intentionally grows phrase memory, counts and alphabet, so a completely flat curve is not expected. The closely matched curves plus weak-reference regressions distinguish legitimate model growth from historical-cache retention.


## Why fully incremental backward messages are deferred

For positional BP,
`beta[t,s] = sum_(s -> d) p(s,d) * mask[t,symbol] * beta[t+1,d]`.
Changing a count generally changes the normalized probabilities of every symbol
in that row, not only the newly observed transition. Its direct influence appears
at every time position where the row is usable. Changes then travel backward
through predecessor states at earlier times. A regular acceptor adds acceptor
state to this dependency, and virtual augmentation can change several transformed
rows for one base update. Order selection and cached sampling candidates also
need invalidation when feasibility or conditional weights change.

The diagnostic rebuilds both tables independently and compares common context
labels with a relative tolerance of `1e-12` (floor `1e-300`). It reports changed
cells, not a claim about the runtime of an incremental algorithm. The Bach
549-note duplicate addition changes only 5/24, 10/136, 15/339 and 19/621 numeric
rows at orders 1–4. Nevertheless **every** common beta value at time zero changes
at each order. At order 4, all 621 cells change for times 0–36; one step before
the end only 10 change. Adding a new symbol shows the same broad propagation on
common states, plus newly introduced contexts. Exact proportional count updates
that leave a distribution unchanged need no numeric update, but identifying that
case alone does not solve broader predecessor propagation.

Updates remain local when masks/acceptor transitions exclude the changed rows,
when changed rows are outside a queried prefix's finite-horizon reachable product,
when a component is disconnected, or when the remaining horizon is short enough
that only a small reverse dependency cone is reachable. The disconnected example
changes 2/4 order-1 start cells and 6/11 order-2 cells, leaving the other component
untouched. The final terminal layer is always unchanged. For a single prefix,
regular BP's existing lazy evaluation already avoids much irrelevant work.

Full recomputation is preferable when the dependency cone covers most of the
relevant time/product graph, when topology or alphabet changes force recompilation,
when constraints/horizon change frequently, or when maintaining reverse indexes,
work queues, numeric/log tables and candidate invalidation costs more than a
linear pass. Graph fallback can also renumber states, invalidating any beta table
indexed by previous IDs. The current short positional recurrence is cheap enough
that Python bookkeeping would need a substantial, demonstrated locality advantage.
Underflow support also requires consistent updates of log-space tables.

Recommendation: adopt immutable count/ordinary-graph reuse; keep full backward
recomputation. Revisit incremental BP only for a measured downstream workload
with mostly stable support, stable request constraints and persistently small
reachable reverse cones. First measure that cone and its bookkeeping cost;
there is no evidence here to justify adding a fully incremental BP engine.

## Validation and integration

The complete suite passes: **198 tests**. Twenty online-learning regressions add
checks for duplicates and empty phrases, START/END boundaries, new symbols,
randomized repeated and deferred updates, terminal/backoff closure, support-shrink
fallback, unchanged row sharing, fresh inference, minimized views, weighted
acceptors, seed-equivalent traces under both built-in policies, regular constraints,
first-hit END and underflow support. The initial investigation also compared
1,000 randomized updates with full rebuilds; the committed tests retain bounded,
seeded versions and the focused orphan-state regression.

Small examples exhaustively enumerate every length-three sequence after a
duplicate addition and a new-symbol addition. An independent rational `Fraction`
recurrence computes the longest-feasible policy probability directly from counts,
without graph or BP tables. Incremental and rebuilt results match every
probability/support value and total mass one, with and without a regular ban.
Virtual snapshots agree with fresh virtual models and explicit augmentation.
Existing exhaustive numerical/positional tests, order/backoff, hard constraints,
regular/MAXORDER/meter, first-hit and log-space tests also run unchanged.

Weak-reference tests show that 100 unprepared updates retain no model chain and
40 prepare/update cycles retain no obsolete models, graphs or BP results once
replacement graphs are prepared. Count and graph sharing is read-only by contract;
mutating the public mappings/rows in place remains unsupported, as already
required for prepared-plan cache validity.

Continuator should keep its encoded symbolic model across phrases, assign the
result of `with_sequences`, and clear plan/backend caches or key them by model
identity plus generation parameters. Update the event decoder for new symbols.
Prepare through the existing model-taking APIs, including first-hit END. Existing
plans deliberately continue to represent their original snapshot; they cannot
silently become inference for the new model. Release them when no longer needed.
The event-taking convenience facades continue to rebuild from full training input.
See the [integration guide](../docs/library_usage.md#online-learning-and-continuator-integration)
and executable [example](../examples/online_learning_backend.py). No changes to
an external Continuator checkout were made.
