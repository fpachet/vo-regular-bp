# Optimization Roadmap

This document collects the optimization ideas discussed for `vo_regular_bp`,
including completed work, experiments that were rejected, small safe next steps,
and larger architectural options.

## September 2026 implementation update

The review-driven implementation added bounded transition-row retention,
cross-order DFA-symbol caching, shared virtual source graphs across horizons,
sequence-only and longest-feasible sampling paths, shared first-hit backward
tables, static product-row sharing, and stable numerical/long-horizon fallbacks.
The candidate-copying and policy-allocation ideas below are implemented for
non-trace calls using the exact built-in `LongestFeasiblePolicy`; generic custom
policies retain their list-based candidate interface. Next-acceptor-state lookup
now reuses the validated DFA-symbol cache.

Terminal/backoff closure in lazy virtual graphs and duration specialization
validation were corrected, with regressions that activate the actual fast path.
Current measured outcomes and remaining tradeoffs supersede the estimates below:
[`implementation results`](../reports/implementation_results_2026_09_07.md).

The main reference workload is the Bach Prelude pitch-only policy-stack setup:

- training length: 592 pitch events
- prefix: first 6 pitch events
- horizon: 32
- MAXORDER copied n-gram constraint: forbidden length 5
- final pitch class: C
- K = 1..6
- policy: `LongestFeasiblePolicy`
- samples: 100
- execution: single-process CPU Python

## Optimization Goals

The hard constraints are:

- preserve exact constraint semantics;
- keep the generic DFA path available;
- avoid heuristic pruning, beam search, or approximate filtering;
- keep paper evaluation scripts available;
- make the library useful for external projects, not only paper benchmarks.

The two main performance regimes are different:

- Paper BP timing cares mostly about the backward pass.
- Library usage often prepares once and samples many continuations, so sampling
  speed and reusable prepared objects matter a lot.

## Current Measured Facts

On the Bach K=6 setup, graph structure across the order stack has substantial
duplicate storage:

| metric | value |
|---|---:|
| fixed-order graph state records | 3587 |
| unique context labels | 1400 |
| duplicate state records | 2187 |
| duplicate state share | 61.0% |
| fixed-order graph edge records | 5246 |
| unique exact transition labels | 3184 |
| duplicate edge records | 2062 |
| duplicate edge share | 39.3% |

However, actual regular BP message states do not overlap exactly across orders
on this benchmark:

| metric | value |
|---|---:|
| sum of time-indexed product states | 22159 |
| unique `(time, context, q)` labels across orders | 22159 |
| exact shared time-product message states | 0 |

This means simple cross-order memo reuse is unlikely to improve BP timing for
the Bach benchmark, even though shared graph storage could reduce memory and
preparation work.

Approximate construction timings measured on the same setup:

| operation | median time |
|---|---:|
| fresh model plus all K=1..6 graphs | 10.7 ms |
| all K=1..6 graphs from a fresh model | 8.8 ms |
| regular masked BP preparation, K=6 | about 21-22 ms |

The current BP profile is dominated by `_RegularBackwardCache.beta`; dict memo
lookup is a visible sub-cost. Current non-trace sampling after the fast path is
about 0.107 ms per generated sequence at K=6.

## Completed Optimizations

### Split Positional Constraints From Regular DFA State

Status: implemented.

Final-C and other purely positional constraints are kept as time-indexed masks
instead of being folded into the regular product acceptor. MAXORDER remains the
regular automaton.

Effect:

- reduces regular state inflation for positional constraints;
- preserves exact semantics;
- improves the Bach MAXORDER + final-C setup because final-C is not part of the
  DFA product state.

This is now part of the high-level `ConstraintSet` compiler: positional
constraints stay positional, while regular constraints are compiled to DFA
acceptors.

### Dense Forbidden-Substring / MAXORDER DFA

Status: implemented.

Forbidden substring constraints can compile to `DenseForbiddenSubstringDFA`.
States are dense integers and transitions are available through
`dense_transition_by_symbol`.

Effect:

- avoids tuple-state and generic transition construction inside the MAXORDER
  acceptor;
- keeps the exact forbidden-substring language;
- benefits both paper MAXORDER experiments and library users with finite
  alphabets.

### Candidate-Set Cache During Sampling

Status: implemented.

`RegularOrderStackBPResult` caches feasible candidate sets keyed by position,
recent history, and acceptor state.

Effect:

- substantially improves repeated sampling from one prepared backend;
- preserves exact sampling semantics because cached candidate sets are derived
  from the same beta tables;
- most useful when drawing many samples from one backend.

### Cumulative-Weight Sampling

Status: implemented.

Candidate sets store cumulative weights so repeated sampling avoids rebuilding
cumulative totals.

Effect:

- reduces per-step sampling overhead;
- exact distribution is unchanged.

### Constraint-Check Hot-Loop Cleanup

Status: implemented.

The regular backward and sampling paths separate the common constraint cases:
no positional constraint, callable constraint, and set-like constraint.

Effect:

- improved BP/backward timing by about 13-15% versus the previous library
  commit in the Bach MAXORDER benchmark;
- product state and edge counts were unchanged.

### Non-Trace Sampling Fast Path

Status: implemented.

`sample()` and `sample_with_orders()` now generate directly instead of routing
through `sample_with_trace()` and allocating an `OrderSampleStep` per emitted
symbol.

Measured effect:

| K | baseline sample ms/seq | fast-path sample ms/seq | speedup |
|---:|---:|---:|---:|
| 1 | 0.1540 | 0.1140 | 1.35x |
| 2 | 0.1263 | 0.0821 | 1.54x |
| 3 | 0.1287 | 0.0837 | 1.54x |
| 4 | 0.1339 | 0.0891 | 1.50x |
| 5 | 0.1410 | 0.0979 | 1.44x |
| 6 | 0.1575 | 0.1072 | 1.47x |

BP timing is unchanged, as expected.

### Prefix-Independent Order-Stack Plans

Status: implemented.

`prepare_constrained_order_stack_plan(...)` prepares the reusable
`model + length + constraints` part of the order-stack backend without binding a
prefix. Callers then use `plan.for_prefix(prefix)` to get the ordinary prepared
backend for a particular prefix.

Effect:

- avoids recompiling constraints and rebuilding reusable backend objects for
  repeated calls with different prefixes;
- positional order-stack plans share fully computed backward tables;
- regular order-stack plans share graph objects and lazy beta memo tables, so
  later prefixes reuse overlapping `(time, context, DFA-state)` messages;
- preserves the existing prefix-based API as a convenience wrapper.

### Experimental ALERGIA Merge Caching

Status: implemented.

The experimental `vo_regular_bp.experimental.alergia_merge(...)` path now
caches work that is repeated during approximate source-state merging:

- projected continuation counts per source state;
- dominant successor state per projected transition label;
- recursive pair-compatibility results within one merge call.

It also tracks active merge classes incrementally:

- class roots are kept in the same order as the previous full-scan
  `_class_roots(...)` helper;
- class members are kept in source-state order;
- after a union, only the two affected member lists and active root list are
  updated.

This is generic and projection-agnostic. A client such as Transformator still
provides the semantics through `symbol_projection` or
`transition_projection(state, symbol, edge)`, while BP remains exact with
respect to the resulting merged source graph.

Local CPU benchmark:

| case | previous median | current median | speedup |
|---|---:|---:|---:|
| Bach order 6, interval projection | 0.169 s | 0.023 s | 7.33x |
| unique self-loop states, 250 states | 0.586 s | 0.063 s | 9.30x |
| unique self-loop states, 400 states | 2.287 s | 0.162 s | 14.12x |
| identical self-loop states, 300 states | 0.189 s | 0.177 s | 1.07x |
| identical self-loop states, 600 states | 0.776 s | 0.721 s | 1.08x |

Interpretation:

- class tracking is the largest win when ALERGIA scans many active classes and
  rejects candidate pairs cheaply;
- projection-heavy sweeps with many statistically compatible states benefit
  from the projected-count and recursive-pair caches;
- workloads where almost every state immediately merges into one class are
  dominated by class-member compatibility checks, so the gain is modest;
- the next generic ALERGIA optimization should probably reduce candidate pairs
  before testing them, for example with coarse signature prefiltering and
  eligible-root filtering.

### LSDB Padded Melody Cumulative-Meter Fast Path

Status: implemented first exact pass; stronger duration-view factoring remains
the next target.

The LSDB pure melody generator uses an exact additive duration constraint with
absorbing PAD termination, optional minimum real-note count, and optional final
real-note acceptance. The generic DFA product path represented this as an
ordinary regular product and spent most preparation time in time-indexed
recursive beta states.

The current exact specialization recognizes the intersection of these
component acceptors by name:

- `padded_melody_duration_total`;
- `final_real_note`;
- `min_real_note_count`.

When this shape is detected, the regular backward cache:

- extracts the PAD symbol, target duration, symbol duration costs, and
  note/rest flags from the component DFAs;
- uses additive suffix tables keyed by remaining slots and remaining duration
  to reject impossible duration/note-count residuals before expanding context
  edges;
- treats completed-duration states as forced PAD suffixes instead of expanding
  arbitrary non-PAD transitions;
- stores compact integer memo keys for the residual meter state;
- groups outgoing context edges by `(duration_cost, is_note)` so all tokens that
  share the same duration viewpoint reuse the same meter feasibility decision.

Measured on the local LSDB Jobim rebuild with `max_order=3`, `761` vocabulary
symbols, `21774` context states, and `43748` context edges:

| case | previous prepare | current prepare |
|---|---:|---:|
| 16 beats / 16 slots / min 8 notes | about 33.2 s | 3.68 s |
| 32 beats / 16 slots / min 8 notes | about 99.1 s | 8.08 s |
| 32 beats / 31 slots / min 14 notes | about 353.2 s | 31.96 s |

Correctness coverage:

- exhaustive tiny-model distribution equality against a manually composed
  generic product DFA;
- full existing test suite;
- LSDB samples checked through the benchmark script with exact total duration,
  legal trailing PAD behavior, and `success_mass=1.0`.

The remaining issue is that the optimization only factors the meter viewpoint
inside each concrete Markov context. It still computes beta over many
`(context, residual_duration, note_count, final_note)` states because symbols
with the same duration may have different next contexts and probabilities.

Next implementation plan:

1. Build a duration-view quotient for each fixed-order context graph. Two
   context states are candidates for the same quotient state when their
   outgoing distributions are equivalent after projecting each edge to
   `(duration_cost, is_note, is_pad)` and quotient destinations.
2. Compute beta on the quotient graph, not on every concrete context, whenever
   the active regular constraint is this additive padded-meter shape.
3. Preserve exact sampling by refining the chosen quotient transition back to
   concrete Markov edges with weights
   `P(symbol | concrete_context) * beta(quotient_dst, residual_after_symbol)`.
   The quotient is only valid if this refinement gives the same beta value as
   the concrete recurrence for every member context.
4. Start with exact partition-refinement/lumping, not approximate merging:
   iteratively refine quotient classes until all duration-view transition
   signatures and destination classes match. If few contexts collapse, abandon
   the quotient for that order and keep the current concrete fast path.
5. Add diagnostics for quotient class count, concrete-to-quotient compression,
   duration-category rows, quotient beta states, quotient fallback rate, and
   refinement time.
6. Validate against the generic path on tiny exhaustive models where same-cost
   symbols lead to different next contexts, because that is the failure mode
   this optimization must avoid.

Diagnostic result, 2026-05-22:

- an exact read-only duration-view quotient diagnostic is now exposed through
  backend diagnostics for this padded melody fast path;
- on the local Jobim 32-beat / 16-slot benchmark, the exact quotient compresses
  `21774` context states to `10666` duration-view classes, a `2.04x` state
  reduction;
- projected meter-relevant graph edges compress from `37332` concrete edges to
  `27755` quotient edges, a `1.35x` projected-edge reduction;
- the strongest order-specific compression is order 3: `14886` states to
  `6394` classes (`2.33x`) and `20972` projected edges to `13559` quotient
  edges (`1.55x`);
- quotient measurement itself took about `0.67 s` on this benchmark.

Conclusion: exact duration-view factoring is real but moderate on this LSDB
graph. It is likely worth a careful prototype only after lower-complexity
remaining-duration/phase cleanup, unless longer horizons show a larger beta
state reduction than the static graph quotient suggests.

### LSDB Virtual Order-Stack Hot Path

Status: implemented.

The LSDB / FlowComposer melody workload using
`VirtualAugmentedOrderStackModel`, 12 chromatic transpositions, string
pitch-duration symbols, and an integrated duration/motion DFA now has a faster
regular order-stack preparation path.

Implemented changes:

- lazy materialization of virtual fixed-order contexts, so unused higher-order
  context sets are not built during prefix-independent plan preparation;
- cached transformed and inverse-transformed symbols inside
  `VirtualAugmentedOrderStackModel`;
- per-DFA-state symbol transition caching in the regular backward cache;
- direct callable transition/weight fast paths for ordinary `DFA` instances;
- beta memo-hit checks inside the edge loop before recursive calls;
- compact internal transition triples instead of per-edge dataclass allocation;
- additive diagnostics for virtual context, augmented count, transition-row,
  beta-cache, and acceptor-symbol cache activity.

Measured on the synthetic LSDB-shaped weighted benchmark, mean preparation time
improved from about `4.23 s` to about `2.66 s` over repeated local runs, with
the same product-state and accepted-transition counts. The benchmark and full
details are in
`reports/lsdb_virtual_order_stack_optimization_2026_05_20.md`.

## Tried And Rejected

### Array-Backed Graph Edges

Status: prototyped and not committed.

Idea:

- replace Python dataclass edge traversal with parallel arrays such as symbols,
  destinations, probabilities, and orders.

Result:

- slower on the Bach MAXORDER benchmark in the tested form.

Likely reason:

- the hot loop still paid Python iteration and indexing costs, while losing some
  locality and clarity from the existing object representation.

Recommendation:

- do not revisit as a local mechanical rewrite;
- only revisit as part of a larger compiled product graph or optional
  accelerated backend.

### Dense Regular Backward Cache

Status: prototyped and removed.

Idea:

- replace `(time, graph_state, acceptor_state)` tuple memo keys with dense
  integer product keys;
- use dense DFA transition tables more directly.

Measured result:

| variant | BP result |
|---|---|
| sparse iterative dense cache | about 0.74x versus current, slower |
| encoded-key recursive dense cache | about 0.86x versus current, slower |

Recommendation:

- do not repeat this as a simple key-shape change;
- any future BP backend should change the overall computation model, not just
  the memo key representation.

## Sampling Optimizations and Remaining Small Options

The first three items were completed in September 2026. Their combined measured
effect is recorded in the implementation report; earlier individual estimates
are not additive performance guarantees.

### Return Cached Candidate Sets Without Copying

Status: implemented for non-trace calls with the exact built-in
`LongestFeasiblePolicy`. Its first-feasible candidate cache returns tuples
directly. Custom policies retain their existing list-based interface.

### Fast Path For `LongestFeasiblePolicy`

Status: implemented. Non-trace sampling stops at the first feasible order and
draws an edge without allocating `PolicyDecision` or `CandidateChoice` objects.
Sequence-only calls also avoid allocating discarded order arrays. Traces and
custom policies retain full candidate evaluation; a first trace can consequently
cost more than a plain sample.

### Reuse Next Acceptor State

Status: implemented through the shared DFA-symbol transition cache. Generic
sampling reuses cached transitions without changing the candidate data shape.

### Prefilter Non-Forbidden Outgoing Edges

Status: optional future experiment.

The code often checks `edge.symbol in forbidden_symbols` in hot loops. For
models with explicit start/end sentinel symbols, prefiltered outgoing lists
could avoid repeated branches.

Expected gain:

- tiny for the Bach setup;
- potentially useful for libraries using sentinel symbols heavily.

Risk:

- low; measure branch costs before adding another retained edge view.

## Bigger Architectural Options

### Exact Quotient / Probabilistic Automaton Minimization

Status: source-graph minimization implemented as an opt-in compiled view;
finite-horizon constrained quotienting is not implemented.

Idea:

- after the count model or reachable BP product is built, compute exact state
  equivalence by partition refinement;
- two rows are equivalent only when they have the same emitted symbols, exact
  probabilities/weights, and successor equivalence classes;
- for constrained finite-horizon BP, compute the quotient time-backwards, since
  positional masks and accepting states make equivalence time-dependent.

Diagnostic on `data/bach_prelude_c_major_pitches.txt`:

| setup | original | exact quotient | reduction |
|---|---:|---:|---:|
| `ContextGraph.from_sequences`, K=6 states | 1401 | 355 | 3.95x |
| `ContextGraph.from_sequences`, K=6 edges | 1834 | 671 | 2.73x |
| order-stack final-C mask, K=6 time states | 32283 | 8275 | 3.90x |
| order-stack final-C mask, K=6 edge relaxations | 37625 | 17940 | 2.10x |
| reachable MAXORDER+final-C product, K=6 time states | 4753 | 743 | 6.40x |
| reachable MAXORDER+final-C product, K=6 positive edges | 10007 | 2768 | 3.62x |

Implementation note:

- the shipped opt-in path is exact source-graph minimization, enabled with
  `minimize_source_graphs=True`;
- fixed-order graph signatures include `edge.order`, so order diagnostics and
  traces keep their semantics;
- minimized graphs are cached on the `OrderStackModel`, so the quotient is not
  rebuilt for every horizon once the model has been warmed.

Initial K=6, horizon-32 Bach timing with `edge.order` preserved:

| setup | minimize | cold prepare ms | warm prepare ms | sample ms/seq | source states | source edges | product time states |
|---|---:|---:|---:|---:|---:|---:|---:|
| final-C only | no | 25.4 | 16.5 | 0.0760 | 3587 | 5246 | n/a |
| final-C only | yes | 45.9 | 12.7 | 0.0776 | 2534 | 4187 | n/a |
| MAXORDER len-5 + final-C | no | 43.6 | 35.3 | 0.1206 | 3587 | 5246 | 24556 |
| MAXORDER len-5 + final-C | yes | 65.4 | 35.2 | 0.1266 | 2534 | 4187 | 24556 |

The opt-in source quotient is therefore useful for memory and warm
positional-only preparation, but it does not by itself reduce the reachable
regular product for this MAXORDER benchmark. The larger constrained-product
reductions in the diagnostic table require a separate horizon-specific quotient
backend.

Expected gain:

- BP/backward: potentially meaningful, especially for constrained finite-horizon
  products where many states have identical constrained futures;
- memory/object count: meaningful;
- sampling: likely helpful if quotient states can share candidate rows.

Risks:

- probabilities are currently normalized floats; exact minimization should
  prefer rational/count-derived values where possible;
- quotienting hides concrete context identity unless representative/context-set
  metadata is preserved for traces and diagnostics;
- online training updates become harder, because count changes can split
  equivalence classes and propagate backward.

Recommendation:

- keep the training/count model uncompressed;
- use the current opt-in source quotient when memory or repeated positional
  preparation matters;
- treat quotient BP/product backends as a secondary experiment, because they are
  tied to a specific horizon and constraint set;
- rebuild the quotient after batch training changes rather than maintaining it
  incrementally.

Real-time Continuator scope:

- source-graph minimization is reusable across calls with different requested
  lengths;
- finite-horizon constrained quotienting is not the first target for real-time
  use, because a final-position constraint makes equivalence depend on the
  current horizon;
- if a small set of horizons recurs, constrained quotients or prepared plans can
  be cached by `(horizon, constraint spec)`.

Future abstraction work:

- an explicit experimental ALERGIA-like source merge is available through
  `vo_regular_bp.experimental.alergia_merge`;
- Continuator-style order stacks can use
  `vo_regular_bp.experimental.alergia_merge_order_stack_model`, which merges
  each fixed-order source graph independently before the ordinary order-stack
  backend is prepared;
- it deliberately merges statistically compatible continuation profiles to
  explore abstraction/generalization capacity;
- clients can provide domain semantics with `symbol_projection` for emitted
  symbols or `transition_projection(state, symbol, edge)` for context-aware
  transition features while the merged graph continues to emit concrete symbols;
- order-stack ALERGIA preserves original context aliases and the existing trace
  shape, but traces describe the explicit merged source model;
- this is a modeling feature, not an exact optimization, and is never enabled
  automatically.

### Compiled Sampling Tables

Status: optional follow-up; re-profile after the September sampling fast paths.

Idea:

- after BP, compile per-step/per-state candidate tables containing selected
  order candidates, cumulative weights, next context, and next acceptor state;
- sampling then becomes table lookup plus RNG.

Expected gain:

- sampling: the earlier 1.3x-2.0x estimate predates the September fast paths
  and must be remeasured against the new baseline;
- prepare time: slightly higher;
- BP: unchanged.

Best for:

- reusable library use;
- external projects that prepare once and sample many continuations;
- Continuator integration.

Risks:

- memory growth if tables are compiled for many states that are never sampled;
- should probably be optional or lazy.

Verification:

- same-seed equivalence against current `sample_with_orders()` for built-in
  policies;
- support equivalence and constraint satisfaction tests;
- distribution checks on tiny enumerable examples.

### Specialized MAXORDER Backend

Status: plausible BP-focused project, not easy.

Idea:

- make forbidden copied n-grams a first-class backend instead of representing
  it only as a generic DFA;
- carry the rolling suffix state directly in the BP transition logic;
- specialize transition lookup and accepting checks for fixed forbidden length.

Expected gain:

- BP/backward: possibly 1.10x-1.30x;
- stretch goal: 1.40x if the specialization avoids enough dict and method-call
  overhead;
- sampling: little direct change.

Best for:

- paper BP-only CPU numbers;
- users who rely heavily on MAXORDER-like constraints.

Risks:

- more backend complexity;
- easy to accidentally change the exact forbidden-substring language;
- the earlier dense-cache experiment shows that naive specialization can
  regress.

Verification:

- exact distribution match against generic DFA/product path on tiny examples;
- support match against brute force;
- Bach benchmark product state/edge counts should remain semantically
  consistent.

### Iterative Sparse Product BP With Precompiled Product Graph

Status: larger redesign. Iterative positional traversal, static product-row
sharing, and iterative evaluation for long regular horizons are implemented;
a compact precompiled regular product backend remains future work.

Idea:

- build compact reachable product transitions once;
- run backward messages iteratively over arrays/lists instead of recursive
  dict memoization;
- reuse the product graph for repeated sampling or repeated beta recomputation.

Expected gain:

- BP/backward: uncertain, maybe 1.15x-1.50x if implemented carefully;
- repeated use: better if product graph is reused;
- prepare time: may increase for one-shot runs.

Best for:

- a reusable library backend with diagnostics and repeated sampling;
- optional high-performance path for dense finite alphabets.

Risks:

- more code and more memory;
- product graph construction could dominate small problems;
- needs careful benchmarking to avoid repeating the array-backed edge
  regression.

### Shared Order-Stack Graph / Suffix DAG

Status: useful for memory and preparation, probably not BP-only speed.

Idea:

- represent all fixed-order contexts in one suffix-indexed structure;
- expose order-specific views without duplicating lower-order context records;
- possibly share exact transition records.

Measured opportunity:

- state records could drop from 3587 to at most 1400 in the Bach K=6 setup;
- exact transition records could drop from 5246 to about 3184;
- graph construction currently costs about 8.8 ms for all K=1..6 graphs.

Expected gain:

- memory/object count: meaningful;
- public preparation time: maybe 10-15%;
- BP/backward: near 0% for the Bach benchmark unless paired with a deeper
  algorithmic redesign.

Risk:

- moderate implementation complexity;
- can obscure the currently simple fixed-order graph semantics.

Recommendation:

- good library architecture project after external integration;
- not the next paper BP optimization.

### Cross-Order Message Reuse

Status: not promising as a direct optimization for Bach MAXORDER.

Idea:

- reuse beta messages for contexts that appear in several order graphs.

Measured issue:

- exact `(time, context, DFA-state)` overlap across orders was zero in the Bach
  K=6 run.

Expected gain:

- BP/backward: likely near 0% for this benchmark;
- may help other corpora or policies, but should be measured first.

Recommendation:

- do not implement as a standalone optimization now.

### Optional NumPy Or Numba Backend

Status: possible future accelerator.

Idea:

- keep the pure-Python exact backend as default;
- add an optional accelerated backend for dense integer symbols/states.

Expected gain:

- potentially the largest CPU gain;
- could exceed 2x on larger dense workloads.

Risks:

- new dependency and packaging complexity;
- harder install path for downstream projects;
- less useful if workloads remain small and sparse.

Recommendation:

- defer until the pure-Python library API has been exercised in Continuator or
  another external project.

### Virtual Data Augmentation

Status: implemented as an investigation backend.

Idea:

- define a finite transformation family, such as integer pitch transpositions;
- keep the generated symbol space absolute;
- compute augmented continuation counts lazily by summing transformed preimages:

```text
C_aug(c -> y) = sum_g C_base(g^-1(c) -> g^-1(y))
```

This preserves the same branch-sharing behavior as explicit data augmentation.
If an absolute context can be explained by multiple transformed copies, all
matching continuation counts are accumulated exactly as if the transformed
sequences had been materialized.

First Bach 12-transposition measurement:

| method | stored events | full graph states | BP graph states | product edges |
|---|---:|---:|---:|---:|
| original corpus | 592 | 3587 | 3587 | 82491 |
| explicit 12-key augmentation | 7104 | 27371 | 27371 | 1968532 |
| virtual 12-key augmentation | 592 | 27371 | 896 | 1968532 |

The virtual backend matches explicit augmentation start masses and success mass
on this setup. The current implementation reduces stored training data and
regular-BP graph materialization, but it does not yet reduce product-edge
expansions for the Bach MAXORDER query. Further gains would require combining
virtual augmentation with a more compact MAXORDER/product representation.

## Priority Recommendation

For the reusable library:

1. Profile full LSDB first-sample expansion and retained symbol/message caches:
   the measured first sample still takes about 13 seconds, and the sampled/traced
   lifecycle peaks at about 1.46 GiB on the reported machine.
2. Use that profile to choose between tighter cache storage and an exact
   duration-view quotient prototype; preserve the generic reference path.
3. Reassess lazy compiled sampling tables against the completed fast paths.
4. Exercise Continuator integration and let real usage guide API changes.
5. Consider a shared suffix graph if cross-order source duplication remains a
   meaningful memory cost after sharing virtual graphs across horizons.

For paper BP-only performance:

1. Do not focus on graph construction first.
2. Consider a specialized MAXORDER backend, but treat it as an experiment with
   strict exactness tests.
3. Avoid simple dense-key rewrites unless a profile shows a new reason.

For broad future performance:

1. Consider a compact precompiled regular product backend if larger corpora
   justify its preparation and memory costs. Long-horizon stack safety is
   already handled by iterative evaluation.
2. Consider optional compiled acceleration only after the library interface has
   stabilized.

## Required Validation For Any New Optimization

Every optimization that can affect sampling or BP should include:

- same success mass or partition mass as the generic path;
- same support as the baseline on tiny enumerable examples;
- same exact distribution on tiny examples, either against brute force or the
  generic BP distribution;
- generated samples satisfy all positional, regular, MAXORDER, and meter
  constraints;
- no changes to the target Bach paper setup unless explicitly requested;
- `python -m pytest -q`;
- benchmark rows with raw runs and medians when the change is performance
  motivated.
