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
object allocation. See [`docs/optimization_roadmap.md`](docs/optimization_roadmap.md)
for measured results, discarded experiments, and future optimization options.

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
returning `None` rejects that symbol from the current acceptor state.

Available acceptor helpers include:

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
- product-state and edge-count diagnostics for scalability studies.

`sample_exact(...)` is a convenience wrapper that runs BP and draws one sample.

### Positional BP

`run_positional_bp(...)` is a no-DFA specialization for fixed-horizon
time-indexed symbol masks. It is useful when the only constraints are
positional, because the recursion ranges over context states rather than full
context-acceptor products.

`LazyBackoffContextModel` matches `ContextGraph.from_backoff_sequences(...)`
semantically, but materializes outgoing edges only when reached by BP.

### Order-Stack BP

`OrderStackModel` and the `run_order_stack_*` functions implement
Continuator-style constrained policy backoff over a stack of fixed-order
models:

- `run_order_stack_bp`: positional constraints only.
- `run_order_stack_dfa_bp`: regular DFA constraints.
- `run_order_stack_masked_dfa_bp`: regular DFA plus positional masks.

This is a generation policy, not exact conditioning of one fixed stochastic
source. At each step, the engine computes feasible future mass for each order,
then an order policy such as `LongestFeasiblePolicy` or
`SingletonAvoidingBackoffPolicy` chooses among feasible candidate orders.

The main result object reports `success_mass`, order-specific start masses,
samples, and optional order traces.

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

For variable-length Continuator-style suffixes, use
`prepare_until_order_stack(...)`. It prepares one fixed-length constrained
backend for each feasible length in `[min_length, max_length]`, samples a
length by the sum of its positive start-order masses, then samples the suffix
from that length backend. If only a backend success indicator is available, the
length receives unit weight. The returned suffix does not include the prefix:
the prefix is conditioning context only.

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
- `regular_acceptors`: caller-supplied `DFA` instances.
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
