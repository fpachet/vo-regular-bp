# Code review — 7 September 2026

This is the baseline review, preserved with its original measurements and
reproductions. The findings below are addressed in the
[implementation results](implementation_results_2026_09_07.md), which document
the completed changes, validation, measured gains, and remaining limitations.

Reviewed local `main` at `4f7c871`, using Python 3.13.11. The checkout was clean before the review. No library code was changed. Scope: public backend, context/product/positional/order-stack BP, virtual augmentation, minimization, tests, and performance documentation. Experimental statistical merging and every paper experiment were not exhaustively audited. Remote CI and remote branch freshness were not checked.

**Objective and status.** This is a dependency-free Python library for constrained symbolic sequence generation. Product BP conditions one stochastic context model on a regular language, optionally with soft transition weights. Order-stack BP implements a separate policy that selects among feasible orders; it is deliberately not conditioning one fixed stochastic source. The music/Continuator/LSDB applications motivate the specialized engines, but the main API supports general hashable symbols.

The project has reusable prefix-independent plans, virtual augmentation, exact source minimization, dynamic soft acceptors, first-hit variable-length generation, examples, and reproducible benchmark scripts. Packaging still declares alpha status, version 0.1.1. The most recent local commit is documentation dated 2 July 2026. The existing suite passed: **83 tests in 2.18 s**. CI is configured for Python 3.10–3.13. These are meaningful strengths, but the boundary cases below need attention before relying on the general exactness claim.

**Confirmed findings, in priority order**

1. **[P1] Lazy virtual graphs discard valid emissions into terminal contexts.** In `vo_regular_bp/augmentation.py:469`, an edge is discarded unless its destination occurs in the context-count index. A valid final event does not need a future continuation row. Training on just `('a', 'b')`, using order 1, prefix `('a',)`, horizon 1, and an identity transform gives start mass **1.0 explicitly and 0.0 virtually**. This directly contradicts explicit/virtual equivalence. Higher-order destinations absent from the count index can also require ordinary suffix backoff. Preserve graph closure and empty terminal rows as the materialized graph does; do not interpret absence from the training-count index as a forbidden transition. Add distribution comparisons involving unique final symbols and backoff destinations.

2. **[P1] Padded-duration specialization assumes more than it verifies.** `vo_regular_bp/order_stack_bp.py:2478` identifies specialized acceptors partly by their display names. At line 2783, additive-duration detection checks only one symbol for each cost. The optimized beta loop at line 1290 then bypasses individual DFA transitions for all symbols in that category. Two equal-duration symbols may have different permissions at a later total. A reproduced example gives mass **1/3 through the generic path and 2/3 through the prepared specialized path**, solely after changing the acceptor name to `padded_melody_duration_total`. Which symbol becomes the representative follows set iteration order. Prefer an explicit, validated specialization contract/builder; otherwise verify all relevant symbols and states and fall back to generic evaluation. Name-based detection also assumes semantics for the final-note and note-count components.

   The current test `tests/test_policy_stack_optimized.py:122`, named `test_padded_melody_fast_path_matches_generic_product_distribution`, does **not activate this fast path**. It calls `run_order_stack_dfa_bp`, whose `_run_order_stack_regular_bp` constructor does not pass a padded specialization. Use the prepared public API and explicitly assert that specialization is active before comparing against the reference. This API divergence is also a maintenance/performance concern.

3. **[P2] Floating-point underflow turns feasible support into failure.** The unscaled recurrence in `vo_regular_bp/product_bp.py:246` and corresponding order-stack recurrences multiplies probabilities and weights directly. A fair IID binary source, horizon 20, and constant weight `1e-20` on every transition produces partition 0, conditional probability 0, and a sampling exception. The conditional source should still be fair IID: the common weight cancels, and the probability of twenty `a` symbols is `2**-20`. Finite weights above one can similarly overflow. Add scaled messages or log-domain messages and a log-partition API; retain a separate feasibility representation where appropriate. Scaling needs careful treatment across orders and variable-length mass comparisons. Merely checking that individual weights are finite is insufficient.

4. **[P2] Linear recursion imposes an undocumented horizon limit.** `_RegularBackwardCache.beta` at `vo_regular_bp/order_stack_bp.py:1070` recurses once per generated symbol. A one-state deterministic model, a true acceptor, and horizon 1,200 raises `RecursionError`, although the state space is trivial. `run_positional_bp` also has a recursive recurrence. Use an explicit traversal stack or an iterative backward pass; raising the process-wide recursion limit is not a robust library fix. This concerns stack safety independently of whether a dense implementation would be faster.

5. **[P2] Virtual source graph caches duplicate horizon-independent structure.** `vo_regular_bp/augmentation.py:344` keys a newly allocated lazy source graph stack by length, but `_new_lazy_graphs` takes neither length nor prefix. Preparing 32 different horizons on one model retains **32 distinct order-1 source graphs**, even after the caller discards the backends. The prefix-specific cache similarly keys independent copies by full prefix and length. Share model/order graph structure, keep horizon-dependent beta tables on plans, and expose cache ownership/reset controls. Cross-plan sharing should use stable state IDs and an immutable training model or explicit invalidation.

**CPU and memory measurements**

The existing LSDB-shaped weighted benchmark was run with its defaults: 120 tracks × 96 events, 12 transforms, maximum order 4, horizon 32. Command:

```sh
.venv/bin/python scripts/bench_lsdb_virtual_order_stack.py --weighted --repeats 3
.venv/bin/python scripts/bench_lsdb_virtual_order_stack.py --weighted --profile --profile-limit 18
```

| Measurement | Observed result |
|---|---:|
| Preparation, three ordinary runs | 2.3823 / 2.4963 / 2.6602 s |
| Median preparation | 2.4963 s |
| Model construction | 0.0257–0.0290 s |
| Beta states | 63,640 |
| Transition rows | 63,391 |
| Accepted transition entries | 1,615,583 |
| Transition-row cache hits during preparation and diagnostics | 1 |
| Traced live Python allocations after preparation | 284.52 MiB |
| Peak traced Python allocations during preparation | 289.80 MiB |
| Row tuple and transition-triple containers alone | 113.35 MiB |

Memory was measured in a separate run with `tracemalloc`, retaining the returned backend. These figures are Python allocation measurements, not process RSS. Of the retained memory, transition triples account for 98.6 MiB; accepted-symbol cache entries and rejected-symbol dictionary storage are also substantial.

Under `cProfile`, preparation took 6.09 s. `accepted_transitions` accounted for 5.17 s inclusive; benchmark DFA transitions were called about 1.56 million times. These profiled times include instrumentation overhead. This identifies transition handling and retained Python objects as stronger targets than model training for this workload.

The benchmark currently returns immediately after preparation and diagnostics (`scripts/bench_lsdb_virtual_order_stack.py:218`). Its one-symbol START prefix initially warms only order 1. In a separate run, the first sample with seed 0 took **15.70 s after tracing was disabled**. An attempted subsequent 100-sample batch was interrupted after the process reached approximately **3.6 GiB RSS**; no completed-batch throughput is claimed. Higher orders and additional beta states are materialized while sampling. Record preparation, first-sample latency, repeated-sample latency, varied-prefix use, and peak memory separately in future benchmarks.

**Highest-value optimization work**

1. **Make transition-row retention selective or optional.** The default benchmark retains over a million transition triples that are almost never reused during preparation. A process-local experiment discarded each completed transition row while keeping beta and acceptor-symbol caches. Preparation times were **1.7350 / 1.7429 / 1.8058 s**, median **1.7429 s**, approximately **30% less time** than the measured baseline. Retained traced allocations fell from **284.52 to 165.37 MiB** (42% less); peak traced allocations were **170.65 MiB**. The start mass remained exactly `1.62386168549132e-08`, with the same 63,640 beta misses. This is evidence for a promising optimization, not a validated universal replacement: workloads with repeated rows and repeated sampling may benefit from retaining them. Benchmark a cache budget, selective retention, or separate policies for preparation and sampling. No implementation was committed.

2. **Share DFA symbol-transition caches across orders within a plan.** Their keys and values depend on the acceptor state and symbol, not source order. Currently each `_RegularBackwardCache` owns a separate copy (`order_stack_bp.py:1009`). This cannot improve the initial order-1-only benchmark much, but it should reduce duplication once sampling touches several orders. Keep caches separate for different soft acceptors and weight versions. Precompute static symbol features such as duration units at the application boundary; the profile repeatedly computes them.

3. **Remove duplicated source graph ownership across horizons**, as described in finding 5. Measure retained bytes rather than context counts alone. The optional minimized graph path keeps the original graph in `_graph_cache` as well as the minimized graph in `_minimized_graph_cache`; fewer active source states therefore do not automatically mean less total retained model memory.

4. **Avoid rebuilding a full DP table for every first-hit length.** `backend.py:854` prepares and retains an independent fixed-length backend per candidate length. For a tiny two-state example, maximum lengths 32, 64, and 128 retain 1,120, 4,288, and 16,768 beta cells respectively: approximately quadratic growth. A shared recurrence indexed by remaining length can make the simple time-homogeneous first-hit case linear in the maximum horizon. Preserve the documented length-weighting policy and retain a generic fallback for horizon-dependent positional/meter constraints.

5. **Add a real sequence-only path for built-in policies.** `sample()` still delegates to `sample_with_orders()`, allocating orders that it discards. `LongestFeasiblePolicy.choose` constructs decision metadata even without tracing; cached candidate tuples are copied to lists. A sequence-only path can avoid those allocations, retain the next DFA state in candidate records, and potentially stop evaluating orders once the first feasible one is found. Preserve the generic custom-policy and trace paths and test RNG behavior. Cumulative candidate weights and bisect sampling already exist, so the next step is not to implement those from scratch.

6. **Keep the generic product graph's static topology separate from time layers.** `product_bp.py:209` recreates `ProductEdge` objects for recurring product states at every time. Sharing immutable transition rows can reduce object count for time-homogeneous acceptors while keeping beta layers separate. Evaluate this on the generic engine independently of the order-stack row-retention experiment: their reuse patterns differ.

For larger changes, consider a shared suffix graph and an optional compiled sparse kernel only after these measurements. Existing documentation records regressions from mechanical array/dense-key rewrites; a new proposal needs end-to-end evidence. Source graph minimization and finite-horizon product minimization have different reuse and memory costs. Approximate ALERGIA merging changes the model and should remain separate from exact optimization work.

**Small reproductions**

The transition-row experiment used this temporary process-local wrapper before running the benchmark's `run_once`; it does not modify source files. For memory measurements the benchmark's backend was retained through a wrapper around its imported `prepare_constrained_order_stack` function, and tracing covered model construction and preparation. Baseline and experimental memory measurements were made separately with clean benchmark parsing/transform caches.

```python
from vo_regular_bp.order_stack_bp import _RegularBackwardCache

original = _RegularBackwardCache.accepted_transitions

def transient_rows(self, state, acceptor_state):
    row = original(self, state, acceptor_state)
    self.transition_rows.pop((state, acceptor_state), None)
    return row

_RegularBackwardCache.accepted_transitions = transient_rows
```

Run from the repository root with its Python interpreter:

```python
from vo_regular_bp import *

# Finding 1: explicit and virtual should have identical mass.
explicit = OrderStackModel.from_sequences([('a', 'b')], max_order=1)
virtual = VirtualAugmentedOrderStackModel.from_sequences(
    [('a', 'b')], max_order=1,
    transforms=(SymbolTransform('identity', lambda s: s, lambda s: s),),
)
for model in (explicit, virtual):
    result = run_order_stack_dfa_bp(
        model, true_acceptor(), length=1, prefix=('a',),
    )
    print(result.start_order_masses())  # ((1, 1.0),), then ((1, 0.0),)

# Finding 3: a common soft weight should cancel from conditioning.
graph = ContextGraph.from_counts({(): {'a': 1, 'b': 1}}, max_order=0)
weighted = WeightedDFA(
    start_state=0, accept_states={0}, transition_func=lambda q, s: 0,
    transition_weight_func=lambda q, s: 1e-20,
)
result = run_bp(graph, weighted, length=20)
print(result.partition_function)  # 0.0
print(result.conditional_probability(('a',) * 20))  # 0.0; should be 2**-20

# Finding 4: trivial graph still exceeds Python's recursion limit.
model = OrderStackModel.from_sequences([('a', 'a')], max_order=1)
run_order_stack_dfa_bp(model, true_acceptor(), length=1200, prefix=('a',))
```

The specialized-duration counterexample is deterministic within each process; it chooses the symbol that is not used as the per-cost representative:

```python
from collections import Counter
from vo_regular_bp import *
from vo_regular_bp.order_stack_bp import _graph_edge_symbols

model = OrderStackModel({
    ('s',): Counter({'a': 1, 'b': 1}),
    ('a',): Counter({'a': 1, 'b': 1, 'P': 1}),
    ('b',): Counter({'a': 1, 'b': 1, 'P': 1}),
    ('P',): Counter({'P': 1}),
}, max_order=1)
symbols = _graph_edge_symbols({1: model.compile_graph(1)})
blocked = [s for s in symbols if s != 'P'][-1]

def transition(state, symbol):
    total, ended = state
    if symbol == 'P':
        return (total, True) if total == 2 else None
    if ended or total == 2 or (total == 1 and symbol == blocked):
        return None
    return (total + 1, False)

for name in ('generic', 'padded_melody_duration_total'):
    dfa = DFA(
        start_state=(0, False),
        states={(t, e) for t in range(3) for e in (False, True)},
        transition_func=transition, accept_func=lambda q: q[0] == 2,
        name=name,
    )
    backend = prepare_constrained_order_stack(
        model, ConstraintSet(regular_acceptors=(dfa,)),
        length=2, prefix=('s',),
    )
    print(name, backend.result.start_order_masses())  # 1/3 versus 2/3
```

Recommended sequence: fix support correctness and fast-path validation first; add real first-sample/memory benchmarks; prototype selective cache retention and source graph sharing; then address numeric stability and iterative traversal. Regression validation should compare support and exact small-example distributions, not just nonzero success indicators or sample constraint satisfaction.
