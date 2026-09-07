# vo-regular-bp

`vo-regular-bp` is a Python library for exact constrained generation from
sparse variable-order context models, with support for positional masks, meter
constraints, forbidden-substring constraints, and reusable order-stack
backends.

The core algorithm runs backward dynamic programming on the reachable product
of a context graph and a deterministic acceptor. Sampling then chooses each next
symbol proportionally to its model probability times the downstream beta value,
so generated sequences are exact samples from the source model conditioned on
the regular language whenever the constrained mass is nonzero.

The repository also contains specialized engines and experiments for positional
constraints and Continuator-style order-stack backoff policies.

## Paper

If you use this library or the accompanying experiments, please cite the
corresponding arXiv paper:

Pachet, F. (2026). *Exact Regular-Constrained Variable-Order Markov Generation
via Sparse Context-State Belief Propagation*. arXiv:2605.07839.
https://doi.org/10.48550/arXiv.2605.07839

```bibtex
@misc{pachet2026exactregularconstrainedvariableorder,
  title = {Exact Regular-Constrained Variable-Order Markov Generation via Sparse Context-State Belief Propagation},
  author = {Pachet, Fran{\c{c}}ois},
  year = {2026},
  eprint = {2605.07839},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  doi = {10.48550/arXiv.2605.07839},
  url = {https://arxiv.org/abs/2605.07839},
}
```

## Install

The package has no declared runtime dependencies and requires Python 3.10 or
newer.

For use from another application, install directly from GitHub:

```bash
python -m pip install "vo-regular-bp @ git+https://github.com/fpachet/vo-regular-bp.git"
```

To pin an application to a stable revision, append a branch, tag, or commit:

```bash
python -m pip install "vo-regular-bp @ git+https://github.com/fpachet/vo-regular-bp.git@main"
```

For local development:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Library-oriented integration notes are in
[`docs/library_usage.md`](docs/library_usage.md). The paper evaluation scripts
are kept separately under [`paper/variable_order_regular_bp/`](paper/variable_order_regular_bp/).
Optimization status and future performance ideas are tracked in
[`docs/optimization_roadmap.md`](docs/optimization_roadmap.md).
Virtual transformed-corpus augmentation is described in
[`docs/virtual_data_augmentation.md`](docs/virtual_data_augmentation.md).

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

## Optimization Status

The library includes several exactness-preserving optimizations: positional
constraints are kept as time masks instead of DFA product state, MAXORDER /
forbidden-substring constraints use a dense DFA when possible, order-stack
sampling caches feasible candidate sets, and non-trace sampling avoids trace
object allocation. The September 2026 update also bounds regular transition-row
retention, shares virtual graphs across horizons, speeds up the built-in
longest-feasible sampling path, and shares first-hit backward tables when
`constraints=None`. Lazy graph closure and duration specialization are covered
by new correctness regressions; stable numerical fallbacks support extreme
masses and long horizons.

Local benchmarks measured 12% less full LSDB preparation time and 40% less
retained traced memory; repeated warm sampling took 58% less time on smaller
LSDB and 74% less on Bach. These gains depend on the workload: cold traces can
take longer, and full LSDB first-sample expansion remains expensive. See the
[`implementation results`](reports/implementation_results_2026_09_07.md) for
parameters, raw measurements, and limitations, and the
[`optimization roadmap`](docs/optimization_roadmap.md) for remaining work.

## Quick Start

```python
from vo_regular_bp import ContextGraph, positional_acceptor, run_bp

graph = ContextGraph.from_counts({(): {"a": 10, "b": 1}}, max_order=0)
acceptor = positional_acceptor(length=1, alphabet=graph.alphabet)

bp = run_bp(graph, acceptor, length=1)

print(bp.partition_function)             # 1.0
print(bp.conditional_probability(("a",))) # 10 / 11
print(bp.sample(rng=0))                   # exact constrained sample
```

Regular constraints can be intersected. This example samples three iid symbols,
forces the final symbol to be `"a"`, and rejects the substring `"bb"`.

```python
from vo_regular_bp import (
    ContextGraph,
    all_of,
    forbidden_substring_acceptor,
    positional_acceptor,
    run_bp,
)

graph = ContextGraph.from_counts({(): {"a": 1, "b": 1}}, max_order=0)
acceptor = all_of(
    positional_acceptor(3, {2: {"a"}}, alphabet=graph.alphabet),
    forbidden_substring_acceptor([("b", "b")], alphabet=graph.alphabet),
)

bp = run_bp(graph, acceptor, length=3)
samples = bp.sample_many(5, rng=123)

assert all(sample[-1] == "a" for sample in samples)
assert all(("b", "b") not in zip(sample, sample[1:]) for sample in samples)
```

## Main Concepts

`ContextGraph` stores a sparse probabilistic context model. It can be built from
explicit counts/probabilities, unweighted or weighted sequences, or an explicit
backoff mixture:

- `ContextGraph.from_counts(...)`
- `ContextGraph.from_probabilities(...)`
- `ContextGraph.from_sequences(...)`
- `ContextGraph.from_weighted_sequences(...)`
- `ContextGraph.from_backoff_sequences(...)`

`DFA` is the deterministic acceptor interface used by product BP. A transition
returning `None` rejects that symbol from the current acceptor state. Acceptors
may also expose `transition_weight(state, symbol) -> float` for soft regular
constraints: BP multiplies the VOMM transition probability by this weight before
normalizing. The default weight is `1.0`; a weight of `0.0` acts as a hard ban.

Available acceptor helpers include:

- `WeightedDFA`: convenience subclass for weighted regular constraints.
- `positional_acceptor`: zero-based per-position allowed symbols.
- `meter_acceptor`: finite meter/class patterns.
- `cumulative_meter_acceptor`: cumulative-cost meter predicates.
- `forbidden_substring_acceptor`: reject any forbidden substring.
- `dense_forbidden_substring_acceptor`: finite-alphabet optimized variant.
- `max_order_acceptor`: reject reference substrings of length `max_order + 1`.
- `all_of`: intersect acceptors.
- `true_acceptor`: accept every finite sequence.

## Engines

### Product BP

`run_bp(graph, acceptor, length=...)` is the general exact engine. It performs
backward DP over reachable `(context_state, acceptor_state)` product states and
returns a `ProductBPResult` with:

- `partition_function`: constrained probability mass.
- `sample(...)` and `sample_many(...)`: exact conditional samples.
- `conditional_probability(sequence)`: probability under the constrained
  distribution.
- `log_partition_function` and `log_conditional_probability(sequence)`:
  logarithmic values for masses outside ordinary floating-point range.
- product-state and edge-count diagnostics for scalability studies.

`sample_exact(...)` is a convenience wrapper that runs BP and draws one sample.

### Source Graph Minimization And Experimental Merging

Exact source minimization is available as an optional compiled view. It is
semantics-preserving and never enabled by default:

```python
from vo_regular_bp import exact_context_graph_quotient_stats, minimize_context_graph

stats = exact_context_graph_quotient_stats(graph)
minimized = minimize_context_graph(graph)
```

Order-stack preparation also accepts `minimize_source_graphs=True`, which
minimizes the source graphs before running the ordinary exact BP backend. The
training/count model remains uncompressed, and minimized fixed-order graphs are
cached on the model for reuse.

Approximate source merging is explicitly experimental and changes the source
model. It lives outside the safe top-level API:

```python
from vo_regular_bp.experimental import alergia_merge

merged = alergia_merge(
    graph,
    alpha=0.01,
    min_support=10,
    recursive=True,
    transition_projection=lambda state, symbol, edge: (
        symbol - state[-1]
        if state and isinstance(state[-1], int) and isinstance(symbol, int)
        else symbol
    ),
)
```

The returned object is still a `ContextGraph`, so `run_bp(...)` remains exact
with respect to the merged source model. This is not constrained-product
minimization and is not applied automatically. The optional
`symbol_projection` and `transition_projection` arguments are where a client
supplies domain semantics for similarity; `transition_projection(state, symbol,
edge)` handles context-relative abstractions and takes precedence when both are
provided. The default compares raw emitted symbols.

Order-stack models can opt into the same experiment by merging each fixed-order
source graph independently, then using the ordinary backend:

```python
from vo_regular_bp.experimental import alergia_merge_order_stack_model

abstract_model = alergia_merge_order_stack_model(
    model,
    alpha=0.01,
    min_support=10,
    recursive=True,
    transition_projection=lambda state, symbol, edge: (
        symbol - state[-1]
        if state and isinstance(state[-1], int) and isinstance(symbol, int)
        else symbol
    ),
)

backend = prepare_constrained_order_stack(
    abstract_model,
    constraints,
    length=8,
    prefix=prefix,
)
```

This preserves the public order-stack sampling path and trace shape. It changes
the source model explicitly: BP and order selection remain exact for the merged
fixed-order graphs that are passed in.

The ALERGIA implementation is optimized for projected comparisons by caching
projected continuation counts, projected successor labels, and recursive
pair-compatibility decisions within one merge call. It also tracks active
merge classes incrementally instead of rebuilding class members during every
candidate comparison. These are internal optimizations only: they do not change
the explicit API or make approximate merging automatic.

### Positional BP

`run_positional_bp(...)` is a no-DFA specialization for fixed-horizon
time-indexed symbol masks. It is useful when the only constraints are
positional, because iterative backward messages range over context states rather
than full context-acceptor products. Its result also exposes logarithmic
partition and conditional probabilities.

`LazyBackoffContextModel` matches `ContextGraph.from_backoff_sequences(...)`
semantically, but materializes outgoing edges only when reached by BP.

### Order-Stack BP

`OrderStackModel` and the `run_order_stack_*` functions implement
Continuator-style constrained policy backoff over a stack of fixed-order
models:

- `run_order_stack_bp`: positional constraints only.
- `run_order_stack_dfa_bp`: regular DFA constraints, including weighted transitions.
- `run_order_stack_masked_dfa_bp`: regular DFA plus positional masks.

This is a generation policy, not exact conditioning of one fixed stochastic
source. At each step, the engine computes feasible future mass for each order,
then an order policy such as `LongestFeasiblePolicy` or
`SingletonAvoidingBackoffPolicy` chooses among feasible candidate orders.

The main result object reports `success_mass`, order-specific start masses,
`start_log_order_masses()`, samples, and optional order traces. Cache ownership,
immutability requirements, and numerical limits are described in the
[usage guide](docs/library_usage.md#cache-lifetime-and-numerical-range).

### Public Constraint Backend

`prepare_constrained_order_stack(...)` is the recommended library-facing entry
point for order-stack generation. It accepts a model and a `ConstraintSet`,
compiles positional constraints as time masks, compiles regular constraints as
acceptors, runs the backend once, and returns a reusable sampler.

```python
from vo_regular_bp import (
    ConstraintSet,
    LongestFeasiblePolicy,
    OrderStackModel,
    prepare_constrained_order_stack,
    prepare_constrained_order_stack_plan,
)

sequence = (60, 64, 67, 72, 76, 67, 71, 72)
model = OrderStackModel.from_sequences([sequence], max_order=2)
final_c = {pitch for pitch in sequence if pitch % 12 == 0}

backend = prepare_constrained_order_stack(
    model,
    ConstraintSet(positional={3: final_c}),
    length=4,
    prefix=sequence[:2],
    policy=LongestFeasiblePolicy(),
)

generated = backend.sample_with_orders(rng=0)
sample = generated.sequence
orders = generated.orders
assert sample[-1] % 12 == 0
```

For convenience, `prepare_constrained_order_stack_from_sequences(...)` builds
the `OrderStackModel` and prepares the backend in one call. The returned
`ConstrainedOrderStackBackend` exposes `sample(...)`, `sample_many(...)`,
`sample_with_orders(...)`, `sample_many_with_orders(...)`, `sample_with_trace(...)`,
and a stable `diagnostics` object with context/product sizes and success mass
when a regular backend is used.

`run_constrained_order_stack(...)` remains available when callers need the raw
internal BP result object.

When the model, horizon, and constraints are fixed but many prefixes will be
sampled, prepare a prefix-independent plan once and bind prefixes later:

```python
plan = prepare_constrained_order_stack_plan(
    model,
    ConstraintSet(positional={3: final_c}),
    length=4,
    policy=LongestFeasiblePolicy(),
)

backend_a = plan.for_prefix(sequence[:2])
backend_b = plan.for_prefix(sequence[2:4])

sample_a = backend_a.sample(rng=0)
sample_b = backend_b.sample(rng=1)
```

For regular constraints, the plan owns the shared graph and backward-message
caches. Binding a prefix warms the lazy beta messages reachable from that prefix,
and later prefixes reuse overlapping cached product states. The old
`prepare_constrained_order_stack(...)` API is now a convenience wrapper around
this plan path when the model supports prefix-independent graph preparation.

For variable-length Continuator-style suffixes, use
`prepare_until_order_stack(...)`. It exposes a fixed-length backend for each
feasible length in `[min_length, max_length]`, samples a length by the sum of its
positive start-order masses, then samples the suffix from that length backend.
With `constraints=None`, these backends share suffix views of one backward
table; additional constraints use separate per-length computation. Length
weights are proportionally rescaled if their absolute masses underflow or
overflow. The returned suffix does not include the prefix: the prefix is
conditioning context only.

```python
from vo_regular_bp import OrderStackModel, prepare_until_order_stack

model = OrderStackModel.from_sequences([("A", "B", "C")], max_order=1)
backend = prepare_until_order_stack(
    model,
    prefix=("A",),
    stop="C",
    min_length=1,
    max_length=3,
)

suffix = backend.sample(rng=0)
assert suffix == ("B", "C")
```

`stop` may be one symbol, an iterable/set of symbols, or a predicate
`stop(symbol) -> bool`. The returned suffix includes the first generated stop
symbol at the final position, and earlier positions are constrained not to
satisfy `stop`. `prepare_until_end_order_stack(..., end_symbol=...)` is a
convenience alias for learned END/sentinel stops. END remains forbidden in
ordinary fixed-length sampling, but first-hit END generation explicitly allows
the sentinel at the final stop position.

`ConstraintSet` supports:

- `positional`: per-time symbol masks or predicates.
- `forbidden_substrings`: exact forbidden substring / MAXORDER constraints,
  compiled to a dense DFA when possible.
- `regular_acceptors`: caller-supplied `DFA` or `WeightedDFA` instances. Soft
  transition weights are multiplied into BP; missing transitions or weight
  `0.0` reject the symbol.
- `meter`: a `MeterConstraint` for finite per-symbol meter/class patterns.
- `cumulative_meter`: a `CumulativeMeterConstraint` for duration/cost
  accumulation, bar-boundary predicates, final total cost, and optional padding
  symbols.

The core API is intentionally not Continuator-specific; adapters for other
projects can map their own event objects to symbols, meter classes, costs, or
regular acceptors.

All high-level constraints are over the generated suffix. Positional indices are
zero-based within the generated sequence, not within the training corpus and not
within `prefix + generated`.

### Constraint Builders

Common constraints can be built and combined without manually constructing
`ConstraintSet` objects:

```python
from vo_regular_bp import (
    combine_constraints,
    final_pitch_class,
    avoid_copied_ngrams,
)

constraints = combine_constraints(
    final_pitch_class(0, length=32),
    avoid_copied_ngrams(reference_pitches, 5),
)
```

Builder helpers include:

- `at_position(...)`
- `final_symbol(...)` and `final_symbols(...)`
- `final_pitch_class(...)`
- `avoid_copied_ngrams(...)`
- `meter_pattern(...)`
- `cumulative_meter(...)`
- `padded_duration_total(...)`
- `combine_constraints(...)`

### Event Adapters

External projects can keep rich event objects at their boundary and encode them
as hashable symbols for the backend:

```python
from dataclasses import dataclass
from vo_regular_bp import (
    EventCodec,
    combine_constraints,
    final_pitch_class,
    prepare_constrained_order_stack_from_events,
)

@dataclass(frozen=True)
class Note:
    pitch: int
    duration: int

codec = EventCodec(
    event_to_symbol=lambda note: (note.pitch, note.duration),
    symbol_to_event=lambda symbol: Note(symbol[0], symbol[1]),
)

backend = prepare_constrained_order_stack_from_events(
    [training_notes],
    combine_constraints(
        final_pitch_class(0, length=8, symbol_to_pitch=lambda symbol: symbol[0]),
    ),
    codec=codec,
    max_order=3,
    length=8,
    prefix=prefix_notes,
)

generated = backend.sample_events_with_orders(rng=0)
print(generated.events)
print(generated.orders)
```

See `examples/event_order_stack_backend.py` for a complete small example.

### Continuator-Style Facade

For projects that want a Continuator-shaped entry point, the dependency-free
facade in `vo_regular_bp.continuator` uses event sequences, a prefix, a horizon,
and constraints:

```python
from vo_regular_bp import (
    combine_constraints,
    duration_total_constraint,
    final_pitch_class_constraint,
    prepare_continuation_backend,
    prepare_continuation_plan,
)

constraints = combine_constraints(
    final_pitch_class_constraint(
        0,
        horizon=8,
        symbol_to_pitch=lambda symbol: symbol[0],
    ),
    duration_total_constraint(
        8,
        horizon=8,
        symbol_to_duration=lambda symbol: symbol[1],
    ),
)

backend = prepare_continuation_backend(
    [training_events],
    prefix=prefix_events,
    horizon=8,
    max_order=4,
    constraints=constraints,
    event_to_symbol=lambda event: (event.pitch, event.duration),
)

generated = backend.sample_events_with_orders(rng=0)
print(generated.events)
print(backend.diagnostics.as_dict())
```

The facade defaults to `SingletonAvoidingBackoffPolicy`, matching the
Continuator-style policy-backoff interpretation. Pass `policy=...` to use a
different order-selection policy. See `examples/continuator_style_backend.py`
for a complete dependency-free example.

When making many calls with the same training material, horizon, and constraints,
use `prepare_continuation_plan(...)` once and bind each prefix later:

```python
plan = prepare_continuation_plan(
    [training_events],
    horizon=8,
    max_order=4,
    constraints=constraints,
    event_to_symbol=lambda event: (event.pitch, event.duration),
)

backend = plan.for_prefix(prefix_events)
generated = backend.sample_events_with_orders(rng=0)
```

For variable musical length in a fixed BP horizon, use
`padded_duration_total_constraint(...)` with explicit zero-duration PAD symbols
added to phrase/bar training sequences via `append_padding(...)`. See
`examples/padded_duration_order_stack_backend.py`.

## Brute Force and Metrics

For small examples, the package includes exact enumeration helpers:

- `brute_force_distribution(...)`
- `brute_force_partition_function(...)`
- `conditional_distribution(...)`

For empirical checks:

- `empirical_distribution(...)`
- `total_variation(...)`

These are intended for tests, toy examples, and exactness validation.

## Paper Experiments

The reusable library lives in `vo_regular_bp/`.  Paper-specific validation and
benchmark implementations are kept under `paper/variable_order_regular_bp/`.
The top-level `scripts/` files remain as compatibility launchers, so existing
commands still work:

```bash
python scripts/eval_tiny_exactness.py
python scripts/eval_scalability.py --help
python scripts/eval_bach_scalability.py --help
python scripts/eval_neurips_ablation.py --help
python scripts/eval_virtual_augmentation.py --help
python scripts/eval_bach_continuator_compare.py --help
python scripts/eval_bach_positional_direct.py --help
```

The Bach data lives in `data/bach_prelude_c_major_pitches.txt`.

Rendered experiment notes are in `reports/`, including:

- `reports/evaluation_report.md`
- `reports/optimization_report.md`
- `reports/bach_contextbp_final_compare.md`
- `reports/bach_policy_backoff_paper_results.md`
- `reports/lsdb_virtual_order_stack_optimization_2026_05_20.md`

## Semantics Note

The direct product-BP path (`ContextGraph` plus `run_bp`) samples exactly from a
single fixed stochastic source conditioned on a regular language:

```text
P_source(x | x in L(A))
```

The order-stack path is exact for a different object: a constrained generation
policy that prefers higher orders when they have positive feasible future mass
and backs off only when needed. Its `success_mass` is therefore a policy success
indicator/mass, not the partition function of one fixed source graph.
