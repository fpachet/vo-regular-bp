# LSDB Virtual Order-Stack Optimization

Date: 2026-05-20

This report documents a VO-side optimization pass for the LSDB / FlowComposer
melody workload. The target case uses `VirtualAugmentedOrderStackModel` with
pitch-duration string tokens such as `"60@0.5"`, 12 chromatic virtual
transpositions, `max_order=4`, a generation length of 32, and one regular
constraint that combines total duration with previous-pitch motion state.

## Benchmark

A reproducible synthetic benchmark was added:

```bash
python scripts/bench_lsdb_virtual_order_stack.py --weighted --repeats 5
```

The benchmark is independent of LSDB but mirrors the important shape:

- 120 synthetic training tracks by default.
- String melody tokens with cached parsing and transposition.
- 12 virtual transposition transforms.
- `max_order=4`.
- `length=32`.
- A total duration target of 16 beats.
- One integrated DFA with duration, range, leap, previous-pitch state, and
  optional transition weights.

Profiling is available with:

```bash
python scripts/bench_lsdb_virtual_order_stack.py --profile --weighted
```

## Baseline

Before this pass, the weighted benchmark showed:

- Mean prepare time: `4.2315s` over 3 runs.
- Profiled prepare time: `12.9359s` under `cProfile`.
- `regular_transition_row_cache_misses=63391`.
- `regular_accepted_transition_count=1615583`.
- Virtual context setup was visible but not dominant; recursive regular BP and
  DFA transition/weight calls dominated the profile.

The regular transition-row cache was almost cold because most rows were unique
for `(graph_state, acceptor_state)`. The better reuse point was `(acceptor_state,
symbol)`, especially with many source graph states sharing outgoing symbols.

## Changes

### Lazy Virtual Contexts

`LazyVirtualFixedOrderContextGraph` no longer eagerly materializes allowed
contexts for every order when a prefix-independent plan is prepared. Each order
materializes its allowed context set only when that graph is first touched.

For the LSDB benchmark prefix `(START,)`, this avoids building order 2-4 virtual
context sets during preparation. The diagnostic
`virtual_context_materialization_calls` dropped from 4 to 1 on the default
benchmark.

### Transform Caching

`VirtualAugmentedOrderStackModel` now caches transformed and inverse-transformed
symbols internally. `augmented_counts` also reuses inverse context transforms.
This keeps semantics equivalent to materialized transformed corpora while
reducing repeated string transposition work.

### Regular BP Hot Path

`_RegularBackwardCache` now includes:

- Per-DFA-state symbol transition caching.
- Direct callable fast paths for ordinary `DFA` instances that use
  `transition_func` and `transition_weight_func`.
- Memo-hit lookup inside the beta edge loop, avoiding a recursive Python call
  when the next beta state is already known.
- Compact internal transition triples instead of allocating `_RegularTransition`
  dataclass instances.

The direct callable path is guarded so custom `DFA` subclasses that override
`next_state` or `transition_weight` keep their existing behavior.

### Diagnostics

`BackendDiagnostics.as_dict()` now includes additional counters:

- `virtual_context_materialization_seconds`
- `virtual_context_materialization_calls`
- `virtual_context_materialization_cache_hits`
- `virtual_context_materialization_cache_misses`
- `augmented_count_calls`
- `augmented_count_cache_hits`
- `augmented_count_cache_misses`
- `virtual_outgoing_row_calls`
- `virtual_outgoing_row_cache_hits`
- `virtual_outgoing_row_cache_misses`
- `regular_transition_rows`
- `regular_transition_row_cache_hits`
- `regular_transition_row_cache_misses`
- `regular_accepted_transitions`
- `regular_beta_state_expansions`
- `regular_beta_cache_hits`
- `regular_beta_cache_misses`
- `regular_acceptor_symbol_transition_cache_hits`
- `regular_acceptor_symbol_transition_cache_misses`

These fields are additive and keep the public API compatible.

## Results

After this pass, the weighted benchmark showed:

- Mean prepare time: `2.5973s` over 3 runs.
- Mean prepare time: `2.6638s` over 5 runs.
- Profiled prepare time: `6.3584s` under `cProfile`.
- `virtual_context_materialization_calls=1`.
- `regular_beta_state_expansions=63640`.
- `regular_beta_cache_hits=1551944`.
- `regular_beta_cache_misses=63640`.
- `regular_acceptor_symbol_transition_cache_hits=1441454`.
- `regular_acceptor_symbol_transition_cache_misses=1559659`.
- `regular_accepted_transition_count=1615583`.

That is roughly a `1.6x` speedup on the measured weighted preparation path,
with equivalent product-state and accepted-transition counts.

## Interpretation

The optimization is worth it for LSDB: the bottleneck is not model training or
count construction, but prefix-bound regular BP warmup. Avoiding eager virtual
context materialization helps, but the largest general win is reducing repeated
DFA transition and beta memo overhead in the hot edge loops.

Further speedups are possible, but likely require a bigger internal redesign:

- A compact iterative beta engine for regular order-stack BP.
- More specialized single-DFA source-row kernels.
- Optional dense encodings for DFA states and source symbols.
- Domain-specific precomputed DFA rows for melody duration/motion constraints.

The current pass stays conservative: no model-order reduction, no change to
virtual augmentation semantics, and no public API break.

## Validation

```bash
pytest -q
python -m py_compile scripts/bench_lsdb_virtual_order_stack.py
python scripts/bench_lsdb_virtual_order_stack.py --weighted --repeats 3
```

`python -m pytest -q` passed: `83 passed`.
